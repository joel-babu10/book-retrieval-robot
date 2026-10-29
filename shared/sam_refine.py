"""
shared/sam_refine.py
----------------------------------------------------------------------------
Optional SAM mask refinement for OBB boxes. Tightens bounding box edges
around spine or book contours using SAM segment predictor.
"""

from pathlib import Path
import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_sam_predictor = None


def get_sam_predictor(verbose=True):
    """Lazy-load SAM predictor for edge refinement."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    ckpt_b = Path(__file__).resolve().parent.parent / "sam_vit_b_01ec64.pth"
    ckpt_h = Path(__file__).resolve().parent.parent / "sam_vit_h_4b8939.pth"

    ckpt_path = None
    model_type = "vit_b"

    if ckpt_b.exists():
        ckpt_path = str(ckpt_b)
        model_type = "vit_b"
    elif ckpt_h.exists():
        ckpt_path = str(ckpt_h)
        model_type = "vit_h"

    if ckpt_path is None:
        if verbose:
            print("[SAM Refine] No SAM checkpoint file found. Edge refinement disabled.", flush=True)
        return None

    try:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            import sys
            venv_site = Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages"
            if venv_site.exists() and str(venv_site) not in sys.path:
                sys.path.insert(0, str(venv_site))
            from segment_anything import sam_model_registry, SamPredictor

        if verbose:
            print(f"[SAM Refine] Loading SAM ({model_type}) from: {ckpt_path}", flush=True)
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        sam.to(device=DEVICE)
        _sam_predictor = SamPredictor(sam)
        if verbose:
            print("[SAM Refine] SAM predictor ready.", flush=True)
        return _sam_predictor
    except Exception as e:
        if verbose:
            print(f"[SAM Refine] Failed to load SAM predictor: {e}. Refinement disabled.", flush=True)
        return None


def preload_sam_refine(verbose=False):
    """Force-load SAM predictor and warm up CUDA memory in background thread."""
    predictor = get_sam_predictor(verbose=verbose)
    if predictor is not None:
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            predictor.set_image(dummy)
        except Exception:
            pass


def order_corners_clockwise(pts):
    """Orders 4 corner points in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.array(pts, dtype=np.float32)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def generate_obb_sam_prompts(corners, img_h, img_w):
    """
    Generates positive prompts along the central axis of the OBB.
    Uses box prompt as primary boundary constraint.
    """
    ordered = order_corners_clockwise(corners)
    tl, tr, br, bl = ordered

    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    left_mid = (tl + bl) / 2.0
    right_mid = (tr + br) / 2.0

    h_axis_len = float(np.linalg.norm(bot_mid - top_mid))
    w_axis_len = float(np.linalg.norm(right_mid - left_mid))

    pos_points = []
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))
    pos_points.append([cx, cy])

    if h_axis_len >= w_axis_len:
        for t in [0.25, 0.75]:
            pt = (1.0 - t) * top_mid + t * bot_mid
            pos_points.append(pt.tolist())
    else:
        for t in [0.25, 0.75]:
            pt = (1.0 - t) * left_mid + t * right_mid
            pos_points.append(pt.tolist())

    point_coords = np.array(pos_points, dtype=np.float32)
    point_labels = np.array([1] * len(pos_points), dtype=np.int32)
    return point_coords, point_labels


def clip_mask_to_obb(mask, corners, img_h, img_w, padding_px=2):
    """
    Intersects SAM mask output with a tight OBB polygon mask.
    Prevents mask bleeding/spilling beyond the detected book boundaries
    or onto adjacent/stacked books.
    """
    obb_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts_int = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(obb_mask, [pts_int], 255)

    if padding_px > 0:
        ksize = padding_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        padded_obb_mask = cv2.dilate(obb_mask, kernel, iterations=1)
    else:
        padded_obb_mask = obb_mask

    clipped = cv2.bitwise_and(mask, padded_obb_mask)
    return clipped


def _trim_solid_book_rect(mask_roi, rect):
    """
    Trims low-density outliers (such as open curled paper flaps or background wall shadows)
    from a book's bounding rectangle so the OBB fits the true physical book body.
    Also enforces vertical shelf spine orthogonality if aspect ratio >= 2.8.
    """
    (cx, cy), (w, h), angle = rect
    if min(w, h) < 20 or max(w, h) < 40 or mask_roi is None or mask_roi.size == 0:
        return rect

    # Determine long axis vs short axis
    is_transposed = (w > h)
    short_dim = h if is_transposed else w
    long_dim = w if is_transposed else h
    rot_angle = angle if not is_transposed else (angle + 90.0)

    # 1. Shelf spine orthogonalization (for vertical shelf books)
    # Shelf spines have aspect ratio >= 2.8 and near-vertical orientation
    norm_angle = (rot_angle + 180.0) % 180.0
    is_vertical_spine = (long_dim / max(short_dim, 1.0) >= 2.8) and (norm_angle <= 25.0 or norm_angle >= 155.0)
    if is_vertical_spine:
        # Snap spine to clean vertical axis (0.0 degrees)
        return ((cx, cy), (short_dim, long_dim), 0.0)

    # 2. Scattered book flap & shadow trimming
    # Rotate mask upright to inspect axis profiles
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    rot_mask = cv2.warpAffine(mask_roi, M, (mask_roi.shape[1], mask_roi.shape[0]))

    half_w = short_dim / 2.0
    half_h = long_dim / 2.0
    x1 = max(0, int(cx - half_w))
    x2 = min(rot_mask.shape[1], int(cx + half_w))
    y1 = max(0, int(cy - half_h))
    y2 = min(rot_mask.shape[0], int(cy + half_h))

    patch = rot_mask[y1:y2, x1:x2]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return rect

    # Profile column density along the short axis (width)
    w_profile = np.mean(patch > 0, axis=0)
    left_solid = 0
    while left_solid < len(w_profile) and w_profile[left_solid] < 0.35:
        left_solid += 1
    right_solid = len(w_profile) - 1
    while right_solid > 0 and w_profile[right_solid] < 0.35:
        right_solid -= 1

    trimmed_w = max(right_solid - left_solid, 1)
    # If a flap or shadow stretched the width by > 8%:
    if trimmed_w < len(w_profile) * 0.92 and trimmed_w >= 30:
        shift_local = (left_solid + right_solid) / 2.0 - len(w_profile) / 2.0
        rad = np.radians(rot_angle)
        new_cx = cx + shift_local * np.cos(rad)
        new_cy = cy + shift_local * np.sin(rad)
        new_short = float(trimmed_w)
        if is_transposed:
            return ((new_cx, new_cy), (long_dim, new_short), angle)
        else:
            return ((new_cx, new_cy), (new_short, long_dim), angle)

    return rect


def extract_book_quad(cnt):
    """
    Extracts the true 4-corner perspective quadrilateral of a book cover
    from its segmented contour. Uses adaptive polygon approximation on the convex hull.
    Falls back to minAreaRect box points only if 4 distinct vertices cannot be formed.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps_factor in np.linspace(0.012, 0.08, 40):
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)


def refine_obb_with_sam(img, corners, return_contour=False):
    """
    Fast ROI-based SAM refinement for a single target book:
    1. Crops a localized Region of Interest (ROI) around the book (2x faster than full frame).
    2. Runs SAM predictor within the ROI with inner cover prompts to prevent wall shadow bleed.
    3. Strictly clips the mask to the detected book polygon.
    4. Extracts exact 4-corner perspective quadrilateral for flat books or vertical orthogonal
       box for shelf spines.
    """
    if corners is None or len(corners) != 4:
        return (corners, None) if return_contour else corners

    predictor = get_sam_predictor()
    if predictor is None:
        return (corners, None) if return_contour else corners

    try:
        img_h, img_w = img.shape[:2]
        corners_f = np.array(corners, dtype=np.float32)

        # 1. Crop localized ROI around the target book with safe padding
        x_min, y_min = np.min(corners_f, axis=0)
        x_max, y_max = np.max(corners_f, axis=0)
        pad = 35
        x0 = max(0, int(x_min - pad))
        y0 = max(0, int(y_min - pad))
        x1 = min(img_w, int(x_max + pad))
        y1 = min(img_h, int(y_max + pad))

        roi = img[y0:y1, x0:x1]
        roi_h, roi_w = roi.shape[:2]
        if roi_h <= 10 or roi_w <= 10:
            return (corners, None) if return_contour else corners

        # Local corners relative to ROI
        local_corners = corners_f - np.array([x0, y0], dtype=np.float32)

        # Set localized ROI in SAM predictor (takes ~400ms vs ~1000ms full frame)
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb_roi)

        # 2. Generate inner positive prompts strictly on the book cover surface
        ordered_loc = order_corners_clockwise(local_corners)
        tl, tr, br, bl = ordered_loc
        cx_loc = float(np.mean(local_corners[:, 0]))
        cy_loc = float(np.mean(local_corners[:, 1]))

        top_mid = (tl + tr) / 2.0
        bot_mid = (bl + br) / 2.0
        left_mid = (tl + bl) / 2.0
        right_mid = (tr + br) / 2.0
        h_axis_len = float(np.linalg.norm(bot_mid - top_mid))
        w_axis_len = float(np.linalg.norm(right_mid - left_mid))

        if h_axis_len >= 1.8 * w_axis_len:
            # Vertical shelf spine: place prompts along the vertical centerline
            pos_points = [
                [cx_loc, cy_loc],
                [cx_loc * 0.85 + top_mid[0] * 0.15, cy_loc * 0.40 + top_mid[1] * 0.60],
                [cx_loc * 0.85 + bot_mid[0] * 0.15, cy_loc * 0.40 + bot_mid[1] * 0.60],
                [cx_loc * 0.70 + tl[0] * 0.15 + bl[0] * 0.15, cy_loc],
                [cx_loc * 0.70 + tr[0] * 0.15 + br[0] * 0.15, cy_loc],
            ]
        else:
            # Flat/scattered book: cover surface prompts
            cover_center = top_mid * 0.55 + bot_mid * 0.45
            pos_points = [
                [cx_loc, cy_loc],
                [cx_loc * 0.70 + tl[0] * 0.30, cy_loc * 0.70 + tl[1] * 0.30],
                [cx_loc * 0.70 + tr[0] * 0.30, cy_loc * 0.70 + tr[1] * 0.30],
                [cover_center[0] * 0.70 + bl[0] * 0.30, cover_center[1] * 0.70 + bl[1] * 0.30],
                [cover_center[0] * 0.70 + br[0] * 0.30, cover_center[1] * 0.70 + br[1] * 0.30],
            ]
        point_coords = np.array(pos_points, dtype=np.float32)
        point_labels = np.ones(len(pos_points), dtype=np.int32)

        box_prompt = np.array([
            max(0.0, np.min(local_corners[:, 0])),
            max(0.0, np.min(local_corners[:, 1])),
            min(float(roi_w), np.max(local_corners[:, 0])),
            min(float(roi_h), np.max(local_corners[:, 1])),
        ])

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_prompt[None, :],
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            return (order_corners_clockwise(corners_f), None) if return_contour else order_corners_clockwise(corners_f)

        # 3. Containment with edge-tolerance: allow SAM to find true outer boundary
        obb_roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        pts_roi_int = local_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(obb_roi_mask, [pts_roi_int], 255)
        # Dilate mask slightly to prevent artificial edge-clipping along book boundaries
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        padded_obb_mask = cv2.dilate(obb_roi_mask, dilate_k)
        ref_area = float(obb_roi_mask.sum())

        ref_aspect = float(max(h_axis_len, w_axis_len) / max(min(h_axis_len, w_axis_len), 1.0))
        target_aspect = ref_aspect if ref_aspect >= 2.2 else 1.45

        best_mask = None
        best_score = -1.0

        for idx, m in enumerate(masks):
            m_uint = (m > 0).astype(np.uint8) * 255
            clipped_m = cv2.bitwise_and(m_uint, padded_obb_mask)
            overlap_area = float(cv2.bitwise_and(m_uint, obb_roi_mask).sum())
            if overlap_area < (0.20 * ref_area):
                continue

            # Measure solidity and aspect ratio of this mask
            m_cnts, _ = cv2.findContours(clipped_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not m_cnts:
                continue
            c_largest = max(m_cnts, key=cv2.contourArea)
            c_area = cv2.contourArea(c_largest)
            (mcx, mcy), (mw, mh), mangle = cv2.minAreaRect(c_largest)
            short_d = max(min(mw, mh), 1.0)
            long_d = max(max(mw, mh), 1.0)
            aspect = long_d / short_d
            box_area = mw * mh
            solidity = c_area / max(box_area, 1.0)
            sam_conf = float(scores[idx]) if idx < len(scores) else 0.8
            aspect_fitness = max(0.0, 1.0 - abs(aspect - target_aspect) / max(target_aspect, 1.0))

            mask_quality = (sam_conf * 0.35) + (solidity * 0.35) + (aspect_fitness * 0.30) + (overlap_area / ref_area) * 0.10

            if mask_quality > best_score:
                best_score = mask_quality
                best_mask = clipped_m

        if best_mask is None:
            best_mask = obb_roi_mask

        # Morphological anti-noise filtering:
        # Close internal holes/glare gaps, remove whisker spikes along edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_roi = max(contours, key=cv2.contourArea) if contours else None

        # 4. Extract edge-refined quadrilateral or orthogonal spine box
        if largest_roi is not None and len(largest_roi) >= 4:
            rect = cv2.minAreaRect(largest_roi)
            (cx_r, cy_r), (w_r, h_r), angle_r = rect
            is_transposed = (w_r > h_r)
            short_dim = h_r if is_transposed else w_r
            long_dim = w_r if is_transposed else h_r
            rot_angle = angle_r if not is_transposed else (angle_r + 90.0)
            norm_angle = (rot_angle + 180.0) % 180.0
            is_vertical_spine = (long_dim / max(short_dim, 1.0) >= 2.5) and (norm_angle <= 25.0 or norm_angle >= 155.0)

            if is_vertical_spine:
                # Upright shelf spine: preserve natural shelf tilt angle (e.g. -3.7 deg)
                refined_local = cv2.boxPoints(((cx_r, cy_r), (short_dim, long_dim), rot_angle))
                refined_corners = order_corners_clockwise(refined_local + np.array([x0, y0], dtype=np.float32))
            else:
                # Flat / scattered book: extract true 4-corner perspective quadrilateral
                quad_local = extract_book_quad(largest_roi)
                refined_corners = order_corners_clockwise(quad_local + np.array([x0, y0], dtype=np.float32))

            # Preserve the true organic SAM segmentation mask contour
            global_contour = largest_roi + np.array([x0, y0], dtype=np.int32)
        else:
            global_contour = pts_roi_int + np.array([x0, y0], dtype=np.int32)
            refined_corners = order_corners_clockwise(corners_f)

        if return_contour:
            return refined_corners, global_contour
        return refined_corners

    except Exception as e:
        print(f"[SAM Refine] Fast ROI refinement warning: {e}", flush=True)
        return (order_corners_clockwise(corners), None) if return_contour else order_corners_clockwise(corners)


def extract_refined_pick_edge(refined_corners, pick_location="mid"):
    """
    Given 4 clockwise-ordered corners [tl, tr, br, bl] of a refined book target:
    Calculates the robot pick edge (top/grasp edge), its pick point, angle, endpoints, and length.
    pick_location: 'mid' (default, edge midpoint) or 'top_left' (top-left edge / corner).
    """
    ordered = order_corners_clockwise(refined_corners)
    tl, tr, br, bl = ordered

    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    length_px = float(np.hypot(dx, dy))

    center = (float(np.mean(ordered[:, 0])), float(np.mean(ordered[:, 1])))

    if pick_location == "top_left":
        nominal_pt = np.array(tl, dtype=float)
    else:
        nominal_pt = np.array((tl + tr) / 2.0, dtype=float)

    # Inset pick point slightly towards the book body/center to prevent
    # background wall silhouette bleed and ensure firm gripper jaw contact
    cx, cy = center
    v_in = np.array([cx - nominal_pt[0], cy - nominal_pt[1]], dtype=float)
    dist_to_center = float(np.hypot(v_in[0], v_in[1]))
    inset_px = min(35.0, max(15.0, dist_to_center * 0.08)) if dist_to_center > 0 else 0.0
    pick_pt = nominal_pt + (v_in / max(dist_to_center, 1e-4)) * inset_px

    d_top = length_px
    d_side = float(np.hypot(br[0] - tr[0], br[1] - tr[1]))
    is_short = (d_top <= d_side)
    normal_deg = float((angle_deg + 90.0 + 180.0) % 360.0 - 180.0)

    return {
        "pick_point_2d": (float(pick_pt[0]), float(pick_pt[1])),
        "pick_edge_endpoints": [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1]))],
        "pick_edge_angle_deg": angle_deg,
        "grasp_approach_angle_deg": normal_deg,
        "pick_edge_length_px": length_px,
        "is_short_edge": is_short,
        "center_2d": center,
    }


def refine_obbs_with_sam(img, corners_list, return_contours=False):
    """Batch refinement wrapper using fast localized ROI prediction."""
    results = [refine_obb_with_sam(img, c, return_contour=return_contours) for c in corners_list]
    return results


def refine_pickup_point_with_sam(img, corners, pick_location="top_left"):
    """
    Fast SAM-based pickup point and grasp edge refinement:
    Instead of segmenting the whole book cover and fitting a 4-corner polygon,
    this focuses SAM specifically on the grasp edge / top-left corner.

    1. Determines top grasp edge and nominal pickup coordinates from YOLO OBB corners.
    2. Crops a tight ROI around the pick edge (not the whole book).
    3. Prompts SAM at the pick zone to extract the precise physical edge line.
    4. Computes sub-pixel pick point, grasp alignment angle, and perpendicular approach vector.
    """
    if corners is None or len(corners) != 4:
        return None

    ordered = order_corners_clockwise(corners)
    tl, tr, br, bl = ordered

    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    yolo_angle = float(np.degrees(np.arctan2(dy, dx)))
    edge_len = float(np.hypot(dx, dy))
    v_line = np.array([dx, dy]) / max(edge_len, 1e-4)

    nominal_pick = tl if pick_location == "top_left" else (tl + tr) / 2.0
    center_2d = (float(np.mean(ordered[:, 0])), float(np.mean(ordered[:, 1])))

    fallback_result = {
        "pick_point_2d": (float(nominal_pick[0]), float(nominal_pick[1])),
        "pick_edge_endpoints": [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1]))],
        "pick_edge_angle_deg": yolo_angle,
        "grasp_approach_angle_deg": float((yolo_angle + 90.0 + 180.0) % 360.0 - 180.0),
        "pick_edge_length_px": edge_len,
        "is_short_edge": (edge_len <= np.hypot(br[0] - tr[0], br[1] - tr[1])),
        "center_2d": center_2d,
        "pick_contour": None,
        "method": "yolo_geometric",
    }

    predictor = get_sam_predictor()
    if predictor is None:
        return fallback_result

    try:
        img_h, img_w = img.shape[:2]

        # Inward prompt point located firmly on book cover near grasp edge
        if pick_location == "top_left":
            pt_prompt = tl + 0.12 * (tr - tl) + 0.08 * (bl - tl)
        else:
            pt_prompt = (tl + tr) / 2.0 + 0.08 * ((bl + br) / 2.0 - (tl + tr) / 2.0)

        # Tight ROI around pick edge only (not whole book)
        cx = (tl[0] + tr[0]) / 2.0
        cy = (tl[1] + tr[1]) / 2.0
        half_box = max(int(edge_len * 0.70), 50)

        x0 = max(0, int(cx - half_box))
        y0 = max(0, int(cy - half_box))
        x1 = min(img_w, int(cx + half_box))
        y1 = min(img_h, int(cy + half_box))

        roi = img[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return fallback_result

        local_prompt = np.array([[pt_prompt[0] - x0, pt_prompt[1] - y0]], dtype=np.float32)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        predictor.set_image(roi_rgb)

        masks, scores, _ = predictor.predict(
            point_coords=local_prompt,
            point_labels=np.array([1]),
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            return fallback_result

        c_largest = max(cnts, key=cv2.contourArea)
        global_cnt = c_largest + np.array([x0, y0], dtype=np.int32)
        cnt_pts = global_cnt.reshape(-1, 2).astype(np.float32)

        # Distance from points to nominal grasp line (tl -> tr)
        n_line = np.array([-v_line[1], v_line[0]])
        dists_to_line = np.abs(np.dot(cnt_pts - tl, n_line))
        edge_pts = cnt_pts[dists_to_line < 25.0]

        if len(edge_pts) >= 5:
            [vx, vy, _, _] = cv2.fitLine(edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            refined_angle = float(np.degrees(np.arctan2(vy[0], vx[0])))
            if np.dot([vx[0], vy[0]], v_line) < 0:
                refined_angle = (refined_angle + 180.0) % 360.0
        else:
            refined_angle = yolo_angle

        # True contact pick point on the physical edge, inset slightly towards center to avoid background wall bleed
        if pick_location == "top_left":
            dists = np.hypot(cnt_pts[:, 0] - tl[0], cnt_pts[:, 1] - tl[1])
            best_pt = cnt_pts[np.argmin(dists)]
        else:
            dists = np.hypot(cnt_pts[:, 0] - nominal_pick[0], cnt_pts[:, 1] - nominal_pick[1])
            best_pt = cnt_pts[np.argmin(dists)]

        cx, cy = center_2d
        v_in = np.array([cx - best_pt[0], cy - best_pt[1]], dtype=float)
        dist_to_c = float(np.hypot(v_in[0], v_in[1]))
        inset_px = min(30.0, max(12.0, dist_to_c * 0.08)) if dist_to_c > 0 else 0.0
        inset_pt = best_pt + (v_in / max(dist_to_c, 1e-4)) * inset_px

        pick_u, pick_v = float(inset_pt[0]), float(inset_pt[1])
        norm_deg = float((refined_angle + 90.0 + 180.0) % 360.0 - 180.0)

        return {
            "pick_point_2d": (pick_u, pick_v),
            "pick_edge_endpoints": [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1]))],
            "pick_edge_angle_deg": refined_angle,
            "grasp_approach_angle_deg": norm_deg,
            "pick_edge_length_px": edge_len,
            "is_short_edge": (edge_len <= np.hypot(br[0] - tr[0], br[1] - tr[1])),
            "center_2d": center_2d,
            "pick_contour": global_cnt,
            "method": "sam_fast_pickup",
        }

    except Exception as e:
        print(f"[SAM Refine] Fast pickup refinement fallback: {e}", flush=True)
        return fallback_result



