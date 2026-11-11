"""
shared/reachable_pick.py
----------------------------------------------------------------------------
Finds a kinematically reachable pick position on a scattered book when
the initial/nominal pick point (e.g. spine mid or center) is unreachable.
----------------------------------------------------------------------------
"""

import numpy as np
import cam2robot as c2r
import dobot_kin as K


def deproject_pixel_safe(intrinsics, u, v, depth):
    """
    Safely converts pixel (u, v) and depth (m) into 3D camera coordinates [x, y, z] (m).
    Works with pyrealsense2 intrinsics, dicts, custom objects, and standard D435 fallback.
    """
    if depth is None or depth <= 0.0:
        return None
    try:
        fx = getattr(intrinsics, "fx", None)
        fy = getattr(intrinsics, "fy", None)
        ppx = getattr(intrinsics, "ppx", None)
        ppy = getattr(intrinsics, "ppy", None)
        if fx is not None and fy is not None and ppx is not None and ppy is not None:
            x = (float(u) - float(ppx)) * float(depth) / float(fx)
            y = (float(v) - float(ppy)) * float(depth) / float(fy)
            return [float(x), float(y), float(depth)]
    except Exception:
        pass

    try:
        import pyrealsense2 as rs
        pt = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth))
        return [float(pt[0]), float(pt[1]), float(pt[2])]
    except Exception:
        pass

    # Safe fallback if intrinsics object is missing or non-standard (e.g. 640x480 D435 stream)
    fx, fy, ppx, ppy = 615.0, 615.0, 320.0, 240.0
    x = (float(u) - ppx) * float(depth) / fx
    y = (float(v) - ppy) * float(depth) / fy
    return [float(x), float(y), float(depth)]


def is_robot_reachable(xyz_cam_m, margin=1.0):
    """
    Checks if a camera-frame 3D coordinate (meters) is kinematically reachable by Dobot.
    Returns (is_reachable: bool, rob_xyz_mm: np.ndarray, reason: str).
    """
    if xyz_cam_m is None or not c2r.is_calibrated():
        return False, None, "No calibration or invalid 3D point"

    rob = c2r.cam_to_robot(xyz_cam_m)
    rx, ry, rz = float(rob[0]), float(rob[1]), float(rob[2])
    ok, reason, _ = K.check(rx, ry, rz, margin=margin)
    return ok, np.array([rx, ry, rz]), reason


def order_corners_clockwise_safe(pts):
    """Orders 4 corner points in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_reachable_scattered_point(
    corners_2d,
    nominal_u,
    nominal_v,
    depth_frame,
    color_intrinsics,
    book_depth=None,
    nominal_xyz_cam=None,
    safety_margin_ratio=0.08,
):
    """
    Checks if nominal pick position is reachable.
    If not reachable, evaluates candidate points across the safe interior
    of the scattered book's bounding quadrilateral to find the best point that:
      1. Lies securely inside the physical book boundary (safe margin from edges)
      2. Is kinematically reachable by Dobot Magician
      3. Minimizes distance to the primary target while maximizing reach margin.

    Returns dict:
      {
        "is_fallback": bool,
        "pick_xyz_m": list [x, y, z],
        "pick_rob_mm": list [rx, ry, rz],
        "pick_uv": (u, v),
        "radius_mm": float,
        "reason": str
      }
    """
    if nominal_xyz_cam is not None:
        ok, rob_xyz, reason = is_robot_reachable(nominal_xyz_cam, margin=1.0)
        if ok:
            r = float(np.hypot(rob_xyz[0], rob_xyz[1]))
            return {
                "is_fallback": False,
                "pick_xyz_m": list(nominal_xyz_cam),
                "pick_rob_mm": list(rob_xyz),
                "pick_uv": (float(nominal_u) if nominal_u is not None else 320.0, float(nominal_v) if nominal_v is not None else 240.0),
                "radius_mm": r,
                "reason": "Nominal pick point is fully reachable",
            }
        print(f"\n[Scattered Reachability] Nominal pick at ({rob_xyz[0]:.1f}, {rob_xyz[1]:.1f}, {rob_xyz[2]:.1f}) mm is unreachable: {reason}", flush=True)
        print(f"[Scattered Reachability] Searching book cover surface for a reachable grasp position...", flush=True)
    elif nominal_u is not None and nominal_v is not None and depth_frame is not None and color_intrinsics is not None:
        from input.depth_utils import deproject_target
        nom_xyz = deproject_target(depth_frame, color_intrinsics, nominal_u, nominal_v, reference_depth=book_depth, corners=corners_2d)
        if nom_xyz is not None:
            ok, rob_xyz, reason = is_robot_reachable(nom_xyz, margin=1.0)
            if ok:
                r = float(np.hypot(rob_xyz[0], rob_xyz[1]))
                return {
                    "is_fallback": False,
                    "pick_xyz_m": list(nom_xyz),
                    "pick_rob_mm": list(rob_xyz),
                    "pick_uv": (float(nominal_u), float(nominal_v)),
                    "radius_mm": r,
                    "reason": "Nominal pick point is fully reachable",
                }
            print(f"\n[Scattered Reachability] Nominal pick at ({rob_xyz[0]:.1f}, {rob_xyz[1]:.1f}, {rob_xyz[2]:.1f}) mm is unreachable: {reason}", flush=True)

    # Reference scalar depth fallback
    ref_d = book_depth
    if ref_d is None and nominal_xyz_cam is not None and len(nominal_xyz_cam) >= 3 and float(nominal_xyz_cam[2]) > 0.05:
        ref_d = float(nominal_xyz_cam[2])
    if ref_d is None or ref_d <= 0.05:
        ref_d = 0.26  # Calibrated table plane depth for D435

    # Check if quadrilateral polygon is available
    poly = None
    if corners_2d is not None:
        try:
            poly = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 2)
            if len(poly) < 4:
                poly = None
        except Exception:
            poly = None

    if poly is None:
        # Radial inward fallback if corners are unavailable
        if nominal_xyz_cam is not None:
            rob_pt = c2r.cam_to_robot(nominal_xyz_cam)
            rx, ry, rz = float(rob_pt[0]), float(rob_pt[1]), float(rob_pt[2])
            r = float(np.hypot(rx, ry))
            if r > 300.0:
                scale = 285.0 / r
                adj_rx, adj_ry = rx * scale, ry * scale
                adj_ok, _, _ = K.check(adj_rx, adj_ry, rz, margin=0.5)
                if adj_ok:
                    R, t = c2r.load()
                    adj_cam_mm = np.linalg.inv(R) @ (np.array([adj_rx, adj_ry, rz]) - t)
                    adj_cam_m = (adj_cam_mm / 1000.0).tolist()
                    return {
                        "is_fallback": True,
                        "pick_xyz_m": adj_cam_m,
                        "pick_rob_mm": [adj_rx, adj_ry, rz],
                        "pick_uv": (float(nominal_u) if nominal_u is not None else 320.0, float(nominal_v) if nominal_v is not None else 240.0),
                        "radius_mm": 285.0,
                        "dist_px": 0.0,
                        "reason": f"Reachable cover pick (X={adj_rx:.1f}, Y={adj_ry:.1f}, Z={rz:.1f}mm, r=285.0mm)",
                    }
        return None

    ordered = order_corners_clockwise_safe(poly)
    tl, tr, br, bl = ordered

    # Sample candidates across safe interior of the book
    # Stay safely within book face
    s_vals = np.linspace(safety_margin_ratio, 1.0 - safety_margin_ratio, 15)
    t_vals = np.linspace(safety_margin_ratio, 1.0 - safety_margin_ratio, 15)

    candidates = []

    # Also sample along edge segments
    for frac in np.linspace(0.12, 0.88, 15):
        pt_e1 = (1.0 - frac) * tl + frac * bl
        candidates.append((float(pt_e1[0]), float(pt_e1[1])))
        pt_e2 = (1.0 - frac) * tl + frac * tr
        candidates.append((float(pt_e2[0]), float(pt_e2[1])))
        pt_e3 = (1.0 - frac) * tr + frac * br
        candidates.append((float(pt_e3[0]), float(pt_e3[1])))
        pt_e4 = (1.0 - frac) * bl + frac * br
        candidates.append((float(pt_e4[0]), float(pt_e4[1])))

    # 2D Grid sampling across cover
    for s in s_vals:
        for t in t_vals:
            pt = (1.0 - s) * (1.0 - t) * tl + s * (1.0 - t) * tr + s * t * br + (1.0 - s) * t * bl
            candidates.append((float(pt[0]), float(pt[1])))

    best_candidate = None
    best_score = float("inf")

    for (cu, cv) in candidates:
        cand_xyz = None
        if depth_frame is not None and color_intrinsics is not None:
            # Query local depth frame if valid
            try:
                w, h = depth_frame.get_width(), depth_frame.get_height()
                iu, iv = int(round(cu)), int(round(cv))
                if 0 <= iu < w and 0 <= iv < h:
                    d = depth_frame.get_distance(iu, iv)
                    if 0.15 <= d <= 0.65 and abs(d - ref_d) <= 0.06:
                        cand_xyz = deproject_pixel_safe(color_intrinsics, cu, cv, d)
            except Exception:
                pass

        if cand_xyz is None and ref_d is not None:
            cand_xyz = deproject_pixel_safe(color_intrinsics, cu, cv, ref_d)

        if cand_xyz is None:
            continue

        rob_pt = c2r.cam_to_robot(cand_xyz)
        rx, ry, rz = float(rob_pt[0]), float(rob_pt[1]), float(rob_pt[2])

        # Test kinematic limits with 1.0 deg safety margin
        reach_ok, why, _ = K.check(rx, ry, rz, margin=1.0)
        if not reach_ok:
            reach_ok, why, _ = K.check(rx, ry, rz, margin=0.0)
            if not reach_ok:
                continue

        r = float(np.hypot(rx, ry))
        # Ideal reachable sweet spot is r ~ 220 to 290 mm
        # Progressive penalty as r gets closer to physical stretch limit (>290mm) or base (<180mm)
        rad_penalty = max(0.0, r - 285.0) * 4.0 + max(0.0, 185.0 - r) * 2.0

        # Distance to nominal pick in image space
        dist_px = float(np.hypot(cu - nominal_u, cv - nominal_v)) if (nominal_u is not None and nominal_v is not None) else 0.0
        score = dist_px * 0.08 + rad_penalty

        if score < best_score:
            best_score = score
            best_candidate = {
                "is_fallback": True,
                "pick_xyz_m": [float(c) for c in cand_xyz],
                "pick_rob_mm": [rx, ry, rz],
                "pick_uv": (cu, cv),
                "radius_mm": r,
                "dist_px": dist_px,
                "reason": f"Reachable cover pick (X={rx:.1f}, Y={ry:.1f}, Z={rz:.1f}mm, r={r:.1f}mm)",
            }

    # If surface candidate search was too restrictive, use radial projection as safety net
    if best_candidate is None and nominal_xyz_cam is not None:
        rob_pt = c2r.cam_to_robot(nominal_xyz_cam)
        rx, ry, rz = float(rob_pt[0]), float(rob_pt[1]), float(rob_pt[2])
        r = float(np.hypot(rx, ry))
        if r > 300.0:
            scale = 285.0 / r
            adj_rx, adj_ry = rx * scale, ry * scale
            adj_ok, _, _ = K.check(adj_rx, adj_ry, rz, margin=0.5)
            if adj_ok:
                R, t = c2r.load()
                adj_cam_mm = np.linalg.inv(R) @ (np.array([adj_rx, adj_ry, rz]) - t)
                adj_cam_m = (adj_cam_mm / 1000.0).tolist()
                best_candidate = {
                    "is_fallback": True,
                    "pick_xyz_m": adj_cam_m,
                    "pick_rob_mm": [adj_rx, adj_ry, rz],
                    "pick_uv": (float(nominal_u) if nominal_u is not None else 320.0, float(nominal_v) if nominal_v is not None else 240.0),
                    "radius_mm": 285.0,
                    "dist_px": 0.0,
                    "reason": f"Reachable cover pick (X={adj_rx:.1f}, Y={adj_ry:.1f}, Z={rz:.1f}mm, r=285.0mm)",
                }

    if best_candidate is not None:
        print(
            f"[Scattered Reachability] Found reachable pick on book surface at pixel "
            f"({best_candidate['pick_uv'][0]:.1f}, {best_candidate['pick_uv'][1]:.1f}) -> "
            f"Robot X={best_candidate['pick_rob_mm'][0]:.1f}, Y={best_candidate['pick_rob_mm'][1]:.1f}, "
            f"Z={best_candidate['pick_rob_mm'][2]:.1f} mm (radius={best_candidate['radius_mm']:.1f} mm, "
            f"shifted {best_candidate['dist_px']:.0f}px from unreachable point).",
            flush=True,
        )
        return best_candidate

    print("[Scattered Reachability] Warning: No candidate point inside book boundary was reachable.", flush=True)
    return None
