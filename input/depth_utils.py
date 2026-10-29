"""
input/depth_utils.py
----------------------------------------------------------------------------
Depth reading + 3D deprojection, shared by spine_pipeline and
scattered_pipeline so this logic lives in exactly one place.

Usage:
    from input.depth_utils import deproject_target

    xyz = deproject_target(depth_frame, color_intrinsics, u, v)
    if xyz:
        x, y, z = xyz  # meters, camera optical frame
"""

import cv2
import numpy as np
import pyrealsense2 as rs

DEFAULT_DEPTH_PATCH_SIZE = 5
MAX_FOREGROUND_DEPTH = 0.46  # Max depth (m) for books on table/shelf; background wall is >= 0.50m
MIN_PHYSICAL_DEPTH = 0.15    # Min valid physical depth (m) to reject near-lens artifacts


def median_depth_patch(depth_frame, u, v, patch_size=DEFAULT_DEPTH_PATCH_SIZE,
                       reference_depth=None, max_valid_depth=MAX_FOREGROUND_DEPTH):
    """
    Returns the robust foreground depth (meters) in a tight 5x5 window around pixel (u, v),
    matching the verify.py sampling logic.
    """
    if depth_frame is None or u is None or v is None:
        return None

    width, height = depth_frame.get_width(), depth_frame.get_height()
    u_int = int(round(u))
    v_int = int(round(v))

    scale = max(1.0, width / 640.0)
    # Tightly bounded window scaled to resolution to stay strictly on spine surface
    for radius in [max(2, int(round(2 * scale))), int(round(4 * scale)), int(round(6 * scale))]:
        depths = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = u_int + dx, v_int + dy
                if 0 <= px < width and 0 <= py < height:
                    d = depth_frame.get_distance(px, py)
                    if MIN_PHYSICAL_DEPTH <= d <= max_valid_depth:
                        if reference_depth is not None and d > (reference_depth + 0.08):
                            continue  # Background behind the book
                        depths.append(d)

        if len(depths) >= 3:
            depths_arr = np.array(depths, dtype=np.float32)
            # Foreground clustering: reject background step jumps (> 8cm from foreground estimate)
            p10 = float(np.percentile(depths_arr, 10))
            fg_depths = depths_arr[depths_arr <= (p10 + 0.08)]
            if len(fg_depths) > 0:
                return float(np.median(fg_depths))

    return None


def extract_robust_book_depth(depth_frame, corners=None, center=None,
                              mask_contour=None, max_depth=MAX_FOREGROUND_DEPTH):
    """
    Extracts the ground-truth physical book surface depth (meters) by sampling
    the interior of the book polygon, eroded from the edges to strictly eliminate
    any silhouette edge bleed from the background wall/shelf.
    """
    if depth_frame is None:
        return None

    try:
        arr = np.asanyarray(depth_frame.get_data()) * 0.001
        h, w = arr.shape
        scale = max(1.0, w / 640.0)

        # 1. Prefer polygon interior (corners or mask contour)
        poly_pts = None
        if mask_contour is not None and len(mask_contour) >= 4:
            poly_pts = mask_contour.astype(np.int32).reshape((-1, 1, 2))
        elif corners is not None and len(corners) == 4:
            poly_pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

        if poly_pts is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_pts], 255)
            # Adaptive erosion proportional to frame resolution (leaving inner 70% of book body)
            k_size = max(5, int(round(5 * scale)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            eroded = cv2.erode(mask, kernel, iterations=1)
            valid = arr[(eroded > 0) & (arr >= MIN_PHYSICAL_DEPTH) & (arr <= max_depth)]
            if len(valid) >= 5:
                p10 = float(np.percentile(valid, 10))
                fg = valid[valid <= (p10 + 0.08)]
                if len(fg) >= 3:
                    return float(np.median(fg))

        # 2. Central axis / center point fallback
        if center is not None:
            cx, cy = int(round(center[0])), int(round(center[1]))
            for r in [int(6 * scale), int(12 * scale), int(18 * scale)]:
                patch = arr[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
                valid = patch[(patch >= MIN_PHYSICAL_DEPTH) & (patch <= max_depth)]
                if len(valid) >= 3:
                    p10 = float(np.percentile(valid, 10))
                    fg = valid[valid <= (p10 + 0.08)]
                    if len(fg) >= 2:
                        return float(np.median(fg))
    except Exception as e:
        print(f"[Depth] Warning extracting robust book depth: {e}", flush=True)

    return None


def deproject_target(depth_frame, color_intrinsics, u, v, patch_size=DEFAULT_DEPTH_PATCH_SIZE,
                     reference_depth=None, corners=None, center=None,
                     max_depth=MAX_FOREGROUND_DEPTH):
    """
    Converts pixel (u, v) into real-world XYZ (meters) in the camera's optical frame.
    Protects against background wall sampling:
      - Uses reference_depth (or extracts robust book depth from corners/center)
      - Rejects background readings (e.g. wall at ~0.51m) that would push Dobot out of reach
      - Falls back seamlessly to true book body depth if local edge pixels have dropouts or bleed
    """
    if depth_frame is None or color_intrinsics is None or u is None or v is None:
        print("[Depth] Invalid arguments provided to deproject_target.", flush=True)
        return None

    # Determine ground-truth book depth if not already supplied
    if reference_depth is None and (corners is not None or center is not None):
        reference_depth = extract_robust_book_depth(
            depth_frame, corners=corners, center=center or (u, v), max_depth=max_depth
        )

    # Sample local depth around (u, v) with background rejection
    local_depth = median_depth_patch(
        depth_frame, u, v, patch_size=patch_size,
        reference_depth=reference_depth, max_valid_depth=max_depth
    )

    # Use local depth if valid; otherwise fallback to reference book surface depth
    final_depth = local_depth
    if final_depth is None or final_depth > max_depth or (reference_depth is not None and final_depth > reference_depth + 0.08):
        if reference_depth is not None:
            final_depth = reference_depth
        else:
            print(f"[Depth] No valid depth at ({u:.0f},{v:.0f}) and no reference depth available.", flush=True)
            return None

    point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(u), float(v)], final_depth)
    x, y, z = point_3d
    src = "local" if (local_depth == final_depth) else "book_body_fallback"
    print(f"[Depth] XYZ = ({x:.3f}, {y:.3f}, {z:.3f}) m  (depth={final_depth:.3f} m, src={src})", flush=True)
    return (float(x), float(y), float(z))
