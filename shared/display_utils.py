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
    cx_m, cy_m, cz_m = None, None, None
    center_xyz = target.get("xyz_center_camera_frame") or target.get("xyz_camera_frame")
    if center_xyz is not None:
        cx_m, cy_m, cz_m = center_xyz
        xyz_str = f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f}) m"
    elif target.get("X") is not None and target.get("Y") is not None and target.get("Z") is not None:
        cx_m, cy_m, cz_m = target["X"], target["Y"], target["Z"]
        xyz_str = f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f}) m"

    # Pick edge coordinates
    pick_u = target.get("top_edge_center_u")
    pick_v = target.get("top_edge_center_v")
    pick_xyz = target.get("xyz_pick_camera_frame") or center_xyz
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

    print(f"\n{prefix}==================== TARGET SUMMARY ====================", flush=True)
    print(f"{prefix}  Title / OCR:   '{title}'{mode_str}", flush=True)
    print(f"{prefix}  Confidence:    Similarity={sim:.2f} | Detector Conf={conf:.2f}", flush=True)
    print(f"{prefix}  Center (u, v): ({u:.1f}, {v:.1f}) px", flush=True)
    print(f"{prefix}  Size & Angle:  {w:.1f} x {h:.1f} px | {angle:.1f}°", flush=True)
    if xyz_str:
        print(f"{prefix}  Center (XYZ):  {xyz_str} (Camera Frame)", flush=True)
    print(f"{prefix}  Geometry:      {geom_str}", flush=True)

    if pick_u is not None and pick_v is not None:
        print(f"{prefix}\n{prefix}----- ROBOT PICK & GRASP COORDINATES -----", flush=True)
        print(f"{prefix}  Pick Point 2D: ({pick_u:.1f}, {pick_v:.1f}) px [Edge Midpoint]", flush=True)
        if pick_xyz is not None:
            px, py, pz = pick_xyz
            print(f"{prefix}  Pick Point 3D: XYZ = ({px:.3f}, {py:.3f}, {pz:.3f}) m", flush=True)
        print(f"{prefix}  Grasp Angle:   {pick_angle:.1f}° [Align Gripper Wrist]", flush=True)
        approach_angle = target.get("grasp_approach_angle_deg")
        if approach_angle is not None:
            print(f"{prefix}  Approach Norm: {approach_angle:.1f}° [Perpendicular Approach]", flush=True)
        if pick_len is not None:
            print(f"{prefix}  Edge Length:   {pick_len:.1f} px", flush=True)
        if endpoints and len(endpoints) == 2:
            p1, p2 = endpoints
            print(f"{prefix}  Edge Endpts:   ({p1[0]:.1f}, {p1[1]:.1f}) -> ({p2[0]:.1f}, {p2[1]:.1f}) px", flush=True)
    print(f"{prefix}========================================================", flush=True)
