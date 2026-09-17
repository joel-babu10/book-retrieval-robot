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

import numpy as np
import pyrealsense2 as rs

DEFAULT_DEPTH_PATCH_SIZE = 5  # 5x5 median patch — robust to noisy/thin single-pixel dropouts


def median_depth_patch(depth_frame, u, v, patch_size=DEFAULT_DEPTH_PATCH_SIZE):
    """
    Returns the median depth (meters) in a patch_size x patch_size window
    around pixel (u, v), skipping zero/invalid readings. More robust than
    a single-pixel read, especially on thin or reflective book edges.

    Returns None if no valid depth reading was found in the patch.
    """
    if depth_frame is None or u is None or v is None:
        return None

    half = patch_size // 2
    depths = []
    width, height = depth_frame.get_width(), depth_frame.get_height()

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px, py = int(round(u)) + dx, int(round(v)) + dy
            if 0 <= px < width and 0 <= py < height:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)

    if not depths:
        return None
    return float(np.median(depths))


def deproject_target(depth_frame, color_intrinsics, u, v, patch_size=DEFAULT_DEPTH_PATCH_SIZE):
    """
    Converts pixel (u, v) + median depth into real-world XYZ (meters) in
    the camera's optical frame, using RealSense intrinsics.

    Returns (x, y, z) or None if no valid depth could be found at that
    pixel even with the patch fallback.
    """
    if depth_frame is None or color_intrinsics is None or u is None or v is None:
        print("[Depth] Invalid arguments provided to deproject_target.", flush=True)
        return None

    depth_m = median_depth_patch(depth_frame, u, v, patch_size=patch_size)
    if depth_m is None:
        print(f"[Depth] No valid depth at ({u:.0f},{v:.0f}) even with patch fallback.", flush=True)
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(u), float(v)], depth_m)
    x, y, z = point_3d
    print(f"[Depth] XYZ = ({x:.3f}, {y:.3f}, {z:.3f}) m  (depth={depth_m:.3f} m)", flush=True)
    return (x, y, z)
