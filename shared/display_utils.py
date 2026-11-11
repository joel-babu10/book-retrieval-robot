"""
shared/display_utils.py
----------------------------------------------------------------------------
Clean, formatted console display utilities.
Suppresses raw numpy array dumps (contours, corners, multi-decimal floats)
from polluting the terminal buffer while maintaining complete precision
in returned dictionary objects.
"""

def print_target_summary(target, prefix=""):
    """
    Prints a concise, human-readable summary of the detected book target
    without dumping raw numpy arrays or hundreds of contour numbers to the terminal.
    """
    if target is None:
        print(f"\n{prefix}[Target] No target detected.", flush=True)
        return

    title = target.get("title") or target.get("text", "Unknown")
    mode = target.get("arrangement_mode") or target.get("pipeline_winner", "")
    mode_str = f" [{mode.upper()}]" if mode else ""
    sim = target.get("similarity", 0.0)
    conf = target.get("conf", 0.0)
    u = target.get("u", 0.0)
    v = target.get("v", 0.0)
    w = target.get("width_px") or target.get("width", 0.0)
    h = target.get("height_px") or target.get("height", 0.0)
    angle = target.get("obb_angle") or target.get("angle", 0.0)

    # 3D coordinates
    xyz_str = ""
    center_xyz = target.get("xyz_center_camera_frame")
    if center_xyz is None:
        center_xyz = target.get("xyz_camera_frame")
    if center_xyz is not None:
        cx_m, cy_m, cz_m = center_xyz
        xyz_str = f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f}) m"
    elif target.get("X") is not None and target.get("Y") is not None and target.get("Z") is not None:
        cx_m, cy_m, cz_m = target["X"], target["Y"], target["Z"]
        xyz_str = f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f}) m"

    # Pick edge coordinates
    pick_u = target.get("top_edge_center_u")
    pick_v = target.get("top_edge_center_v")
    pick_xyz = target.get("xyz_pick_camera_frame")
    if pick_xyz is None:
        pick_xyz = center_xyz
    pick_angle = target.get("pick_edge_angle_deg")
    if pick_angle is None:
        pick_angle = angle
    pick_len = target.get("pick_edge_length_px")
    endpoints = target.get("pick_edge_endpoints")

    has_corners = target.get("corners") is not None
    has_mask = target.get("mask_contour") is not None
    geom = []
    if has_corners:
        geom.append("OBB corners verified")
    if has_mask:
        geom.append("SAM mask edge-refined")
    geom_str = ", ".join(geom) if geom else "standard"

    print(f"\n{prefix}Target Summary: '{title}'{mode_str}", flush=True)
    print(f"{prefix}  Title / OCR:   '{title}'{mode_str}", flush=True)
    print(f"{prefix}  Confidence:    Similarity={sim:.2f} | Detector Conf={conf:.2f}", flush=True)
    print(f"{prefix}  Center (u, v): ({u:.1f}, {v:.1f}) px", flush=True)
    print(f"{prefix}  Size & Angle:  {w:.1f} x {h:.1f} px | {angle:.1f}°", flush=True)
    if xyz_str:
        print(f"{prefix}  Center (XYZ):  {xyz_str} (Camera Frame)", flush=True)
    print(f"{prefix}  Geometry:      {geom_str}", flush=True)

    if pick_u is not None and pick_v is not None:
        print(f"{prefix}Robot Pick & Grasp Coordinates:", flush=True)
        print(f"{prefix}  Pick Point 2D: ({pick_u:.1f}, {pick_v:.1f}) px [Edge Midpoint]", flush=True)
        if pick_xyz is not None:
            px, py, pz = pick_xyz
            print(f"{prefix}  Pick Point 3D: XYZ = ({px:.3f}, {py:.3f}, {pz:.3f}) m", flush=True)
        print(f"{prefix}  Grasp Angle:   {pick_angle:.1f}° [Align Gripper Wrist]", flush=True)
        if target.get("is_front_facing"):
            try:
                from robot.dobot_controller import compute_front_facing_rotation
                import cam2robot as c2r
                if pick_xyz is not None and c2r.is_calibrated():
                    r_rob = c2r.cam_to_robot(np.array(pick_xyz) * 1000.0)
                    r_front = compute_front_facing_rotation(r_rob[0], r_rob[1], pick_angle)
                    print(f"{prefix}  Dobot Wrist R: {r_front:+.1f}° [Front-Facing Tangential Grasp]", flush=True)
            except Exception:
                pass
        approach_angle = target.get("grasp_approach_angle_deg")
        if approach_angle is not None:
            print(f"{prefix}  Approach Norm: {approach_angle:.1f}° [Perpendicular Approach]", flush=True)
        if pick_len is not None:
            print(f"{prefix}  Edge Length:   {pick_len:.1f} px", flush=True)
        if endpoints and len(endpoints) == 2:
            p1, p2 = endpoints
            print(f"{prefix}  Edge Endpts:   ({p1[0]:.1f}, {p1[1]:.1f}) -> ({p2[0]:.1f}, {p2[1]:.1f}) px", flush=True)



import cv2
import numpy as np


def draw_corner_brackets(img, pts, color=(246, 165, 96), length=18, thickness=2):
    pts = np.asarray(pts, float).reshape((-1, 2))
    if pts.shape[0] < 4:
        return
    for i in range(4):
        p1 = pts[i]
        p_next = pts[(i + 1) % 4]
        p_prev = pts[(i - 1) % 4]
        v1 = p_next - p1
        v2 = p_prev - p1
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 > 0:
            u1 = v1 / d1
            q1 = (p1 + u1 * min(length, d1 * 0.35)).astype(int)
            cv2.line(img, tuple(p1.astype(int)), tuple(q1), color, thickness, cv2.LINE_AA)
        if d2 > 0:
            u2 = v2 / d2
            q2 = (p1 + u2 * min(length, d2 * 0.35)).astype(int)
            cv2.line(img, tuple(p1.astype(int)), tuple(q2), color, thickness, cv2.LINE_AA)


def draw_aesthetic_target_overlay(img, target):
    """
    Renders a clean, aesthetic, modern AR overlay on the image:
      1. Soft translucent tint over target book body
      2. Sleek thin contour + luminous corner brackets cyan-blue
      3. Precision circular reticle at pick / grasp point
      4. Minimal floating dark badge pill with title and match %
      NO clutter, NO ugly giant black terminal dump box!
    """
    if target is None or img is None:
        return img

    res = img.copy()
    h, w = res.shape[:2]

    # Extract corners
    corners = target.get('corners_refined')
    if corners is None:
        corners = target.get('corners')
    if corners is None:
        corners = target.get('obb_corners')
    if corners is None:
        corners = target.get('box_corners')

    corners_arr = None
    if corners is not None:
        try:
            c_test = np.asarray(corners, dtype=np.float32).reshape((-1, 2))
            if c_test.shape[0] == 4:
                corners_arr = c_test
                pts_int = corners_arr.astype(np.int32).reshape((-1, 1, 2))

                # 1. Subtle soft tint (cyan-blue glow, alpha=0.12)
                overlay = res.copy()
                cv2.fillPoly(overlay, [pts_int], (235, 130, 59))
                res = cv2.addWeighted(overlay, 0.12, res, 0.88, 0)

                # 2. Thin sleek outline + bold corner brackets
                cv2.polylines(res, [pts_int], isClosed=True, color=(246, 165, 96), thickness=1, lineType=cv2.LINE_AA)
                draw_corner_brackets(res, corners_arr, color=(246, 165, 96), length=18, thickness=2)
        except Exception:
            corners_arr = None

    # 3. Precision Pick Reticle
    pick_pt = target.get('pick_point_2d') or target.get('pick_point_px') or target.get('pick_px')
    if pick_pt is None and 'top_edge_center_u' in target and 'top_edge_center_v' in target:
        pick_pt = (target['top_edge_center_u'], target['top_edge_center_v'])
    if pick_pt is None and 'top_edge_center' in target:
        pick_pt = target['top_edge_center']
    if pick_pt is None and 'u' in target and 'v' in target:
        pick_pt = (target['u'], target['v'])
    if pick_pt is None and 'center' in target:
        pick_pt = target['center']

    if pick_pt is not None:
        try:
            pt_arr = np.asarray(pick_pt, float).flatten()
            if len(pt_arr) >= 2:
                px, py = int(round(pt_arr[0])), int(round(pt_arr[1]))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(res, (px, py), 9, (246, 165, 96), 1, cv2.LINE_AA)
                    cv2.circle(res, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.line(res, (px - 14, py), (px - 10, py), (246, 165, 96), 1, cv2.LINE_AA)
                    cv2.line(res, (px + 10, py), (px + 14, py), (246, 165, 96), 1, cv2.LINE_AA)
                    cv2.line(res, (px, py - 14), (px, py - 10), (246, 165, 96), 1, cv2.LINE_AA)
                    cv2.line(res, (px, py + 10), (px, py + 14), (246, 165, 96), 1, cv2.LINE_AA)
        except Exception:
            pass

    # 4. Floating Pill Badge Above Book
    title = target.get('title') or target.get('text', 'TARGET')
    sim = target.get('sim') if target.get('sim') is not None else target.get('similarity', 1.0)
    sim_pct = f'{int(sim * 100)}%' if sim <= 1.0 else f'{int(sim)}%'
    disp_title = title if len(title) <= 22 else title[:20] + '…'
    label = f'{disp_title}  ·  {sim_pct}'

    if corners_arr is not None:
        top_y = int(np.min(corners_arr[:, 1]))
        center_x = int(np.mean(corners_arr[:, 0]))
    elif pick_pt is not None:
        pt_arr = np.asarray(pick_pt, float).flatten()
        top_y = int(pt_arr[1])
        center_x = int(pt_arr[0])
    else:
        top_y, center_x = 80, w // 2

    badge_y = max(34, top_y - 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

    pad_x, pad_y = 10, 6
    bx1 = max(10, min(center_x - (tw // 2) - pad_x, w - tw - 2 * pad_x - 10))
    by1 = badge_y - th - pad_y
    bx2 = bx1 + tw + 2 * pad_x
    by2 = badge_y + pad_y

    pill_overlay = res.copy()
    cv2.rectangle(pill_overlay, (bx1, by1), (bx2, by2), (10, 12, 18), -1)
    res = cv2.addWeighted(pill_overlay, 0.85, res, 0.15, 0)
    cv2.rectangle(res, (bx1, by1), (bx2, by2), (246, 165, 96), 1, cv2.LINE_AA)
    cv2.putText(res, label, (bx1 + pad_x, by2 - pad_y), font, font_scale, (241, 244, 249), font_thick, cv2.LINE_AA)

    if badge_y < top_y:
        cv2.line(res, (center_x, by2), (center_x, top_y), (246, 165, 96), 1, cv2.LINE_AA)
        cv2.circle(res, (center_x, top_y), 2, (246, 165, 96), -1)

    return res
