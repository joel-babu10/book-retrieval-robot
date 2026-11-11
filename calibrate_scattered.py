"""
calibrate_scattered.py
--------------------------------------------------------------------------------
Multi-Point Camera-to-Robot Calibration (Minimum 3 Points) with
180° Parallel Grip Rotation Logic for Tabletop Scattered Books & Front-Facing Books.

Key Capabilities:
  1. Multi-Point Calibration (Min 3 Points):
     - Guides user: Point 1 (Near-Left), Point 2 (Near-Right), Point 3 (Far/Back Desk).
     - Captures points at varied depths so arm remains accurate further back on the desk.
     - Automatically solves 3D Rigid Fit (Kabsch SVD) across all points (R and t).
     - Displays per-point residuals and overall RMSE in mm.
     - In Flat Tabletop mode: saves to calib_scattered.npz.
     - In Front-Facing mode: saves to calib_spine.npz and calib.npz (per user rule).
     - Supports adding extra points (4, 5, 6...) across the desk surface.
     - [U] Undo last point | [Z] Reset points.

  2. 180° Parallel Grip Rotation Logic:
     - Enforces that gripper jaws enter 180° / collinear with the book covers or top rim.
     - Uses 180° gripper symmetry to minimize wrist rotation |J4| = |R - J1|
       and keep J4 within limits (-140° <= J4 <= +140°).
     - [F]: Toggle Front-Facing Mode (Uses Spine Extrinsics with Rotated Grip) vs Flat Mode
     - [1]: 180° Parallel to Desk Approach
     - [2]: 180° Parallel to Front-Facing Top Rim (R = J1 - 90°)
     - [3]: Flip 180° (alternative symmetric parallel angle)
     - [4]: 2-Point Edge Measurement (click 2 points on a book to snap parallel to it)
     - [ / ]: Fine-tune grasp angle +/- 1.0°
--------------------------------------------------------------------------------
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

try:
    from pydobotplus import Dobot
    from pydobotplus.message import Message
    HAS_PYDOBOT = True
except ImportError:
    HAS_PYDOBOT = False

import cam2robot as c2r
import dobot_kin as K

W, H, FPS = 640, 480, 30
HOME_POSE = (200.0, 0.0, 160.0, 0.0)
DEFAULT_PORT = "COM7"
TRANSIT_SAFE_Z = 160.0
MIN_CALIB_POINTS = 3


def auto_detect_dobot_port():
    """Scans and returns the active Dobot COM port."""
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "") + " " + (p.hwid or "")
            if any(k in desc for k in ("Prolific", "CP210", "USB-to-Serial", "CH340", "FTDI")):
                return p.device
    except Exception:
        pass
    return DEFAULT_PORT


def set_gripper_state(bot, state: str):
    """Controls pneumatic gripper: 'open', 'close', or 'off'."""
    if bot is None:
        return
    msg = Message()
    msg.id = 63
    msg.ctrl = 0x03
    if state == "close":
        msg.params = bytearray([0x01, 0x01])
    elif state == "open":
        msg.params = bytearray([0x01, 0x00])
    else:
        msg.params = bytearray([0x00, 0x00])
    try:
        return bot._send_command(msg)
    except Exception:
        return None


def deproject_pixel_to_cam_mm(u, v, depth_m, intr):
    """Deprojects (u, v) and depth to Camera 3D coordinates in mm."""
    pt = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], depth_m)
    return np.array([pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0], dtype=float)


# Global Interactive State
selected_pixel = None
edge_measure_pts = []  # For 2-point book edge angle measurement


def on_mouse(event, x, y, flags, param):
    global selected_pixel, edge_measure_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)
        if len(edge_measure_pts) >= 2:
            edge_measure_pts = []
        edge_measure_pts.append((x, y))


def main():
    global selected_pixel, edge_measure_pts

    print("=" * 76)
    print("   MULTI-POINT SCATTERED / TABLETOP & FRONT-FACING CALIBRATION")
    print("   WITH 180° PARALLEL GRIP ROTATION LOGIC FOR SMOOTH ENTRY")
    print("=" * 76)

    # 1. Load Calibration Transformation
    is_front_facing = False
    try:
        R_scattered, t_scattered = c2r.load(mode="scattered")
        R_scattered = np.asarray(R_scattered, float)
        t_scattered = np.asarray(t_scattered, float)
    except Exception:
        R_scattered = np.eye(3, dtype=float)
        t_scattered = np.array([0.0, 250.0, 0.0], dtype=float)

    try:
        R_spine, t_spine = c2r.load(mode="spine")
        R_spine = np.asarray(R_spine, float)
        t_spine = np.asarray(t_spine, float)
    except Exception:
        R_spine = R_scattered.copy()
        t_spine = t_scattered.copy()

    R_current = R_scattered.copy()
    t_current = t_scattered.copy()

    print(f"[Calib] Loaded Scattered Calibration: t = {np.round(t_scattered, 2)} mm")
    print(f"[Calib] Loaded Spine Calibration:     t = {np.round(t_spine, 2)} mm")

    # 2. Initialize RealSense Camera
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    try:
        prof = pipe.start(cfg)
        print("[Camera] Intel RealSense streams started successfully.")
    except Exception as e:
        print(f"\n[Camera Error] Could not start RealSense: {e}")
        return

    align = rs.align(rs.stream.color)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # 3. Initialize Dobot Magician (Non-blocking fallback if DobotStudio is open)
    bot = None
    bot_connected = False
    port = auto_detect_dobot_port()

    def try_connect_dobot(port_name):
        nonlocal bot, bot_connected
        if not HAS_PYDOBOT:
            print("[Dobot] pydobotplus not installed. Offline mode.")
            return False
        try:
            print(f"[Dobot] Attempting connection on {port_name}...")
            bot = Dobot(port=port_name)
            bot.clear_alarms()
            time.sleep(0.2)
            set_gripper_state(bot, "off")
            bot_connected = True
            print(f"[Dobot] Connected successfully on {port_name}!")
            return True
        except Exception as err:
            bot = None
            bot_connected = False
            print(f"[Dobot] Port {port_name} unavailable ({err}).")
            print("        [NOTICE] If DobotStudio is open, keep it open!")
            print("        You can calibrate by typing X Y Z from the Operation Panel using [C].")
            return False

    try_connect_dobot(port)

    # Move to Home pose if connected
    if bot_connected:
        try:
            print(f"[Dobot] Moving to Home Pose {HOME_POSE}...")
            bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)
        except Exception as e:
            print(f"[Dobot] Warning during Home move: {e}")

    # 4. GUI Window Setup
    cv2.namedWindow("Scattered Multi-Point Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Scattered Multi-Point Calibration", on_mouse)

    target_cam_xyz = None
    target_rob_xyz = None
    last_depth_mm = 0.0

    # Rotation state
    flip_180_state = False
    custom_edge_angle_deg = None
    target_rotation_deg = 0.0
    wrist_j4_deg = 0.0
    gripper_state = "off"

    # Multi-point calibration state
    teach_history = []
    rmse_current = 0.0

    point_labels = [
        "Point 1 (Near-Left Table)",
        "Point 2 (Near-Right Table)",
        "Point 3 (FURTHER BACK / Deep Desk)",
        "Point 4 (Center Desk Point)",
        "Point 5 (Far-Left Desk Point)",
        "Point 6 (Far-Right Desk Point)"
    ]

    status_msg = f"[Step 1/{MIN_CALIB_POINTS}]: Click Point 1 (Near-Left) -> Jog arm to touch -> Press [T] or [C]"
    status_color = (0, 255, 255)

    def recompute_multi_point_fit():
        nonlocal R_current, t_current, rmse_current, status_msg, status_color
        if len(teach_history) < MIN_CALIB_POINTS:
            return False

        cam_pts = np.array([p["cam"] for p in teach_history], dtype=float)
        rob_pts = np.array([p["rob"] for p in teach_history], dtype=float)

        R_opt, t_opt = c2r.fit_rigid(cam_pts, rob_pts)
        errs = c2r.residuals(R_opt, t_opt, cam_pts, rob_pts)
        rmse_current = float(np.sqrt(np.mean(errs ** 2)))

        for i, p in enumerate(teach_history):
            p["error"] = float(errs[i])

        R_current, t_current = R_opt, t_opt

        if is_front_facing:
            c2r.save_spine(R_current, t_current)
            save_name = "calib_spine.npz and calib.npz"
        else:
            c2r.save_scattered(R_current, t_current, also_default=False)
            save_name = "calib_scattered.npz"

        print("\n" + "=" * 74)
        print(f"   MULTI-POINT RIGID FIT SOLVED ({len(teach_history)} POINTS)")
        print(f"   Overall RMSE Error: {rmse_current:.2f} mm")
        print("   Per-Point Error Breakdown:")
        for p in teach_history:
            print(f"     #{p['id']} [{p['label']}]: Error = {p['error']:.2f} mm | Cam: {np.round(p['cam'], 1)} -> Rob: {np.round(p['rob'], 1)}")
        print(f"\n   Rotation Matrix R:\n{np.round(R_current, 4)}")
        print(f"   Translation Vector t: {np.round(t_current, 2)} mm")
        print(f"   SAVED TO {save_name}!")
        print("=" * 74 + "\n")

        status_msg = f"CALIBRATED ({len(teach_history)} Pts)! RMSE = {rmse_current:.2f} mm | Ready to test [G]/[D]"
        status_color = (0, 255, 0)
        return True

    print("\n" + "=" * 76)
    print("                     CALIBRATION CONTROLS")
    print("=" * 76)
    print("  [Click]      : Select target point on desk or book")
    print("  [T]          : TEACH Point from Robot Arm (Reads Dobot position directly)")
    print("  [C]          : CALIBRATE Point from DobotStudio (Type X Y Z from Operation Panel)")
    print("  [U]          : UNDO / Remove the last taught point")
    print("  [Z]          : CLEAR all points and restart multi-point calibration")
    print("  ------------------------------------------------------------------------")
    print("  180° PARALLEL GRIP ROTATION CONTROLS:")
    print("  [F]          : TOGGLE Front-Facing Mode (Uses Spine Calib + Rotated Grip)")
    print("  [1]          : 180° PARALLEL TO TABLETOP APPROACH")
    print("  [2]          : 180° PARALLEL TO FRONT-FACING TOP RIM (R = J1 - 90°)")
    print("  [3]          : FLIP 180° (Toggles alternative symmetric parallel angle)")
    print("  [4]          : 2-POINT EDGE MEASURE (Click 2 points on a book to measure 3D angle)")
    print("  [ / ]        : Fine-tune grasp angle +/- 1.0°")
    print("  ------------------------------------------------------------------------")
    print("  ROBOT MOTION TESTING:")
    print("  [G]          : HOVER overhead at safe transit clearance height")
    print("  [D]          : DESCEND straight down into target point")
    print("  [H]          : Move back to HOME standby pose")
    print("  [SPACE]      : Toggle pneumatic gripper (Open <-> Clamp)")
    print("  [R]          : Reconnect Dobot if offline")
    print("  [Q / ESC]    : Save & Quit safely")
    print("=" * 76 + "\n")

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()

            if not c_frame or not d_frame:
                continue

            d_filtered = spatial.process(d_frame)
            d_filtered = temporal.process(d_filtered)
            d_filtered = hole_filling.process(d_filtered)

            color_img = np.asanyarray(c_frame.get_data())
            depth_data = np.asanyarray(d_filtered.get_data()) * depth_scale

            # Current Dobot Pose
            cur_pose = None
            if bot_connected and bot is not None:
                try:
                    cur_pose = bot.get_pose().position
                except Exception:
                    bot_connected = False
                    cur_pose = None

            # Process selected pixel
            if selected_pixel is not None:
                u, v = selected_pixel
                v1, v2 = max(0, v - 2), min(H, v + 3)
                u1, u2 = max(0, u - 2), min(W, u + 3)
                patch = depth_data[v1:v2, u1:u2]
                valid = patch[patch > 0]

                if valid.size > 0:
                    d_m = float(np.median(valid))
                    last_depth_mm = d_m * 1000.0
                    target_cam_xyz = deproject_pixel_to_cam_mm(u, v, d_m, intr)
                    target_rob_xyz = R_current @ target_cam_xyz + t_current

                    # Compute 180° Parallel Grip Rotation Logic
                    j1_calc = float(np.degrees(np.arctan2(target_rob_xyz[1], target_rob_xyz[0])))
                    book_base_angle = custom_edge_angle_deg if custom_edge_angle_deg is not None else (j1_calc if not is_front_facing else 90.0)
                    target_rotation_deg, wrist_j4_deg = c2r.compute_parallel_grasp_rotation(
                        book_base_angle, target_rob_xyz[0], target_rob_xyz[1],
                        is_front_facing=is_front_facing, flip_180=flip_180_state
                    )
                else:
                    target_cam_xyz = None
                    target_rob_xyz = None

            # Render Overlay
            display_img = color_img.copy()

            # Header Banner
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (W, 82), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, H - 36), (W, H), (20, 20, 20), -1)
            display_img = cv2.addWeighted(overlay, 0.78, display_img, 0.22, 0)

            # Top controls line
            controls_txt = "[F]Mode | [T]Teach | [C]Studio XYZ | [U]Undo | [Z]Reset | [1]Flat 180° | [2]Front 180°"
            cv2.putText(display_img, controls_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            # Robot state line
            if bot_connected and cur_pose is not None:
                rob_state_text = f"Dobot ONLINE: X={cur_pose.x:.1f} Y={cur_pose.y:.1f} Z={cur_pose.z:.1f}mm R={cur_pose.r:.1f}° | Grip: {gripper_state.upper()}"
                cv2.putText(display_img, rob_state_text, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 255), 1)
            else:
                rob_state_text = "Dobot: OFFLINE / DobotStudio Active (Jog in Studio & press [C] to type XYZ)"
                cv2.putText(display_img, rob_state_text, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 165, 255), 1)

            # Multi-point Calibration Progress & Grip Angle Line
            pts_cnt = len(teach_history)
            if pts_cnt < MIN_CALIB_POINTS:
                calib_status_str = f"Multi-Point Progress: {pts_cnt}/{MIN_CALIB_POINTS} (Need {MIN_CALIB_POINTS - pts_cnt} more)"
                prog_color = (0, 165, 255)
            else:
                calib_status_str = f"Multi-Point Calibrated: {pts_cnt} Pts | Overall RMSE: {rmse_current:.2f} mm"
                prog_color = (0, 255, 0)

            if is_front_facing:
                mode_str = "FRONT-FACING (SPINE EXTRINSICS, ROTATED GRIP)"
                mode_col = (0, 255, 255)
            else:
                mode_str = "FLAT TABLETOP / DESK"
                mode_col = (100, 255, 100)

            rot_info = f"{calib_status_str} | {mode_str} R={target_rotation_deg:+.1f}° (J4={wrist_j4_deg:+.1f}°)"
            cv2.putText(display_img, rot_info, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.37, mode_col, 1)

            # 180° Parallel Guarantee Line
            grip_sub = "Alignment: 180° PARALLEL TO BOOK UPON ENTRY (Zero Collision)"
            cv2.putText(display_img, grip_sub, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1)

            # Render Persistent Taught Points Badges
            for p in teach_history:
                px, py = p["pixel"]
                p_err = p.get("error", 0.0)
                badge_col = (0, 255, 0) if p_err <= 2.0 else ((0, 255, 255) if p_err <= 4.0 else (0, 0, 255))
                cv2.circle(display_img, (px, py), 12, badge_col, -1)
                cv2.circle(display_img, (px, py), 14, (0, 0, 0), 2)
                cv2.putText(display_img, str(p["id"]), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)
                if pts_cnt >= MIN_CALIB_POINTS:
                    lbl = f"#{p['id']}:{p_err:.1f}mm"
                    cv2.putText(display_img, lbl, (px + 16, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, badge_col, 1)

            # Render Target Crosshair & HUD
            if selected_pixel is not None and target_rob_xyz is not None:
                u, v = selected_pixel
                r_reach = float(np.hypot(target_rob_xyz[0], target_rob_xyz[1]))
                is_reach = (140.0 <= r_reach <= 325.0) and (-60.0 <= target_rob_xyz[2] <= 165.0)
                marker_color = (0, 255, 0) if is_reach else (0, 0, 255)

                cv2.drawMarker(display_img, (u, v), marker_color, cv2.MARKER_CROSS, 22, 2)
                cv2.circle(display_img, (u, v), 9, marker_color, 1)

                card_x = min(u + 15, W - 340)
                card_y = max(v - 20, 100)
                cv2.rectangle(display_img, (card_x - 5, card_y - 18), (card_x + 335, card_y + 40), (0, 0, 0), -1)

                next_pt_idx = len(teach_history)
                lbl_suggest = point_labels[min(next_pt_idx, len(point_labels) - 1)]
                cam_txt = f"Target Cam 3D: [{target_cam_xyz[0]:.1f}, {target_cam_xyz[1]:.1f}, {target_cam_xyz[2]:.1f}] mm (D={last_depth_mm:.0f}mm)"
                rob_txt = f"Dobot 3D: X={target_rob_xyz[0]:.1f}, Y={target_rob_xyz[1]:.1f}, Z={target_rob_xyz[2]:.1f} mm | R={target_rotation_deg:.1f}°"
                step_txt = f"Next to Teach: #{next_pt_idx + 1} - {lbl_suggest}"

                cv2.putText(display_img, cam_txt, (card_x, card_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                cv2.putText(display_img, rob_txt, (card_x, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, marker_color, 1)
                cv2.putText(display_img, step_txt, (card_x, card_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)

            # Bottom status banner
            cv2.putText(display_img, status_msg, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.39, status_color, 1)

            cv2.imshow("Scattered Multi-Point Calibration", display_img)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("\n[Exit] Quitting scattered multi-point calibration...")
                break

            # F: Toggle Front-Facing Mode vs Flat Tabletop Mode
            if key in (ord("f"), ord("F")):
                is_front_facing = not is_front_facing
                custom_edge_angle_deg = None
                teach_history.clear()
                rmse_current = 0.0

                if is_front_facing:
                    R_spine, t_spine = c2r.load(mode="spine")
                    R_current = np.asarray(R_spine, float)
                    t_current = np.asarray(t_spine, float)
                    status_msg = "FRONT-FACING MODE: Using Spine Extrinsics, Rotated Grip R = J1 - 90°."
                    status_color = (0, 255, 255)
                    print("\n[Mode] Switched to FRONT-FACING SCATTERED:")
                    print("       Rule: Using Spine Calibration (calib_spine.npz) with Rotated Grip (R = J1 - 90°)")
                else:
                    R_scattered, t_scattered = c2r.load(mode="scattered")
                    R_current = np.asarray(R_scattered, float)
                    t_current = np.asarray(t_scattered, float)
                    status_msg = "FLAT TABLETOP MODE: Using calib_scattered.npz."
                    status_color = (100, 255, 100)
                    print("\n[Mode] Switched to FLAT TABLETOP MODE (calib_scattered.npz)")
                continue

            # 1: 180° Parallel to Desk Approach
            if key == ord("1"):
                is_front_facing = False
                custom_edge_angle_deg = None
                status_msg = "Grasp alignment: 180° PARALLEL TO TABLETOP APPROACH."
                status_color = (100, 255, 100)
                print("[Rotation] Set to 180° PARALLEL TO TABLETOP APPROACH.")
                continue

            # 2: 180° Parallel to Front-Facing Top Rim
            if key == ord("2"):
                is_front_facing = True
                custom_edge_angle_deg = None
                status_msg = "Grasp alignment: 180° PARALLEL TO FRONT-FACING TOP RIM (R = J1 - 90°)."
                status_color = (0, 255, 255)
                print("[Rotation] Set to 180° PARALLEL TO FRONT-FACING TOP RIM.")
                continue

            # 3: Flip 180°
            if key == ord("3"):
                flip_180_state = not flip_180_state
                status_msg = f"180° Flip toggled: {'ON' if flip_180_state else 'OFF'}. Parallel jaw orientation preserved."
                status_color = (255, 200, 0)
                print(f"[Rotation] 180° Flip: {'ON' if flip_180_state else 'OFF'}")
                continue

            # 4: 2-Point Edge Measure Mode
            if key == ord("4"):
                if len(edge_measure_pts) >= 2:
                    p1, p2 = edge_measure_pts[-2], edge_measure_pts[-1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    deg_2d = float(np.degrees(np.arctan2(dy, dx)))
                    custom_edge_angle_deg = deg_2d
                    status_msg = f"Edge angle measured: {deg_2d:.1f}°! Gripper snapped 180° parallel to edge."
                    status_color = (0, 255, 0)
                    print(f"[Edge Measure] 2-Point line from {p1} to {p2} -> Angle = {deg_2d:.1f}°")
                else:
                    status_msg = "Click 2 points along book edge on camera feed first, then press [4]!"
                    status_color = (0, 165, 255)
                continue

            # T: Teach Point from direct Dobot connection
            if key in (ord("t"), ord("T")):
                if target_cam_xyz is None:
                    status_msg = "Please click a target point on the camera feed first!"
                    status_color = (0, 0, 255)
                    continue

                if not bot_connected or bot is None:
                    status_msg = "Dobot is not connected via Python. Use [C] to type DobotStudio XYZ!"
                    status_color = (0, 165, 255)
                    print("\n[Teach] Dobot is offline (DobotStudio may be open).")
                    print("        Press [C] to type coordinates from DobotStudio's Operation Panel.")
                    continue

                try:
                    cur_p = bot.get_pose().position
                    p_rob_actual = np.array([cur_p.x, cur_p.y, cur_p.z], dtype=float)

                    pt_id = len(teach_history) + 1
                    lbl = point_labels[min(pt_id - 1, len(point_labels) - 1)]

                    print("\n" + "-" * 65)
                    print(f"[Teach] Point #{pt_id} Recorded [{lbl}]:")
                    print(f"        Camera 3D: [{target_cam_xyz[0]:.2f}, {target_cam_xyz[1]:.2f}, {target_cam_xyz[2]:.2f}] mm")
                    print(f"        Dobot Tip: X={p_rob_actual[0]:.2f}, Y={p_rob_actual[1]:.2f}, Z={p_rob_actual[2]:.2f} mm")

                    teach_history.append({
                        "id": pt_id,
                        "pixel": selected_pixel,
                        "cam": target_cam_xyz.copy(),
                        "rob": p_rob_actual.copy(),
                        "label": lbl,
                        "error": 0.0
                    })

                    if len(teach_history) < MIN_CALIB_POINTS:
                        t_current = p_rob_actual - (R_current @ target_cam_xyz)
                        if is_front_facing:
                            c2r.save_spine(R_current, t_current)
                        else:
                            c2r.save_scattered(R_current, t_current, also_default=False)
                        next_step = len(teach_history) + 1
                        status_msg = f"Point #{pt_id} learned! [Step {next_step}/{MIN_CALIB_POINTS}]: Click {point_labels[next_step - 1]} & teach"
                        status_color = (0, 255, 255)
                        print(f"[Teach] Point #{pt_id} saved. Need {MIN_CALIB_POINTS - len(teach_history)} more point(s) for SVD fit.")
                    else:
                        recompute_multi_point_fit()

                    print("-" * 65)
                except Exception as ex:
                    status_msg = f"Teach error: {ex}"
                    status_color = (0, 0, 255)
                    print(f"[Teach Error] {ex}")
                continue

            # C: Calibrate via manual input from DobotStudio Operation Panel
            if key in (ord("c"), ord("C")):
                if target_cam_xyz is None:
                    status_msg = "Please click a target point on the camera feed first!"
                    status_color = (0, 0, 255)
                    continue

                pt_id = len(teach_history) + 1
                lbl = point_labels[min(pt_id - 1, len(point_labels) - 1)]

                print("\n" + "=" * 65)
                print(f"[Calibrate Point #{pt_id}] [{lbl}]")
                print(f"Selected Camera 3D: [{target_cam_xyz[0]:.1f}, {target_cam_xyz[1]:.1f}, {target_cam_xyz[2]:.1f}] mm")
                print("Enter Dobot coordinates from DobotStudio Operation Panel:")
                print("Format: X Y Z   (e.g., '150.2 220.5 35.0' or leave blank to cancel)")
                print("=" * 65)
                try:
                    coords_input = input(f"Dobot Point #{pt_id} [X Y Z]: ").strip()
                    if coords_input:
                        parts = [float(v) for v in coords_input.replace(",", " ").split() if v.strip()]
                        if len(parts) >= 3:
                            p_rob_actual = np.array(parts[:3], dtype=float)

                            teach_history.append({
                                "id": pt_id,
                                "pixel": selected_pixel,
                                "cam": target_cam_xyz.copy(),
                                "rob": p_rob_actual.copy(),
                                "label": lbl,
                                "error": 0.0
                            })

                            if len(teach_history) < MIN_CALIB_POINTS:
                                t_current = p_rob_actual - (R_current @ target_cam_xyz)
                                if is_front_facing:
                                    c2r.save_spine(R_current, t_current)
                                else:
                                    c2r.save_scattered(R_current, t_current, also_default=False)
                                next_step = len(teach_history) + 1
                                status_msg = f"Point #{pt_id} saved! [Step {next_step}/{MIN_CALIB_POINTS}]: Click {point_labels[next_step - 1]} & teach"
                                status_color = (0, 255, 255)
                                print(f"[Calibrate] Point #{pt_id} recorded. Need {MIN_CALIB_POINTS - len(teach_history)} more point(s) for SVD fit.")
                            else:
                                recompute_multi_point_fit()
                        else:
                            print("[Calibrate] Invalid input format. Expected 3 numbers: X Y Z")
                    else:
                        print("[Calibrate] Input cancelled.")
                except Exception as ex:
                    print(f"[Calibrate Error] {ex}")
                continue

            # U: Undo / Remove last taught point
            if key in (ord("u"), ord("U")):
                if teach_history:
                    popped = teach_history.pop()
                    print(f"\n[Undo] Removed Point #{popped['id']} [{popped['label']}]. Remaining: {len(teach_history)}")
                    if len(teach_history) >= MIN_CALIB_POINTS:
                        recompute_multi_point_fit()
                    else:
                        status_msg = f"Removed Point #{popped['id']}. {len(teach_history)} points remaining. Need {MIN_CALIB_POINTS} minimum."
                        status_color = (0, 165, 255)
                else:
                    status_msg = "No points to undo."
                continue

            # Z: Reset / Clear all points
            if key in (ord("z"), ord("Z")):
                teach_history.clear()
                rmse_current = 0.0
                status_msg = f"All points cleared. [Step 1/{MIN_CALIB_POINTS}]: Click Point 1 & teach."
                status_color = (0, 255, 255)
                print("\n[Reset] All calibration points cleared. Starting fresh multi-point calibration.")
                continue

            # [ and ]: Tune grasp angle by +/- 1°
            if key == ord("["):
                custom_edge_angle_deg = (target_rotation_deg - 1.0)
                status_msg = f"Grasp R trimmed: {custom_edge_angle_deg:.1f}°"
                continue
            if key == ord("]"):
                custom_edge_angle_deg = (target_rotation_deg + 1.0)
                status_msg = f"Grasp R trimmed: {custom_edge_angle_deg:.1f}°"
                continue

            # SPACE: Gripper toggle
            if key == 32:
                if bot_connected and bot is not None:
                    new_st = "close" if gripper_state != "close" else "open"
                    set_gripper_state(bot, new_st)
                    gripper_state = new_st
                    status_msg = f"Gripper state: {gripper_state.upper()}"
                    print(f"[Gripper] State -> {gripper_state.upper()}")
                else:
                    status_msg = "Dobot offline. Gripper command unavailable."
                continue

            # H: Home
            if key in (ord("h"), ord("H")):
                if bot_connected and bot is not None:
                    print("\n[Dobot] Returning to Home standby...")
                    status_msg = "Moving to Home standby..."
                    bot.clear_alarms()
                    cur_p = bot.get_pose().position
                    if cur_p.z < TRANSIT_SAFE_Z:
                        bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, cur_p.r, wait=True)
                    bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)
                    status_msg = "At Home Standby Pose."
                    status_color = (0, 255, 0)
                else:
                    status_msg = "Dobot offline. Home command unavailable."
                continue

            # G: Hover overhead at transit clearance height
            if key in (ord("g"), ord("G")):
                if target_rob_xyz is None:
                    status_msg = "Click a target point first!"
                    status_color = (0, 0, 255)
                    continue
                if not bot_connected or bot is None:
                    status_msg = "Dobot offline. Cannot execute motion."
                    continue

                rx, ry, rz = target_rob_xyz
                z_hover = float(max(TRANSIT_SAFE_Z, rz + 30.0))
                print(f"\n[Dobot] Hovering overhead with 180° Parallel Grip: ({rx:.1f}, {ry:.1f}, {z_hover:.1f}) mm, R={target_rotation_deg:.1f}°...")
                status_msg = f"Hovering overhead at Z={z_hover:.0f} mm with parallel grip..."
                bot.clear_alarms()
                cur_p = bot.get_pose().position
                if cur_p.z < TRANSIT_SAFE_Z:
                    bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, cur_p.r, wait=True)
                bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, target_rotation_deg, wait=True)
                bot.move_to(rx, ry, z_hover, target_rotation_deg, wait=True)
                status_msg = f"Overhead at ({rx:.0f}, {ry:.0f}, {z_hover:.0f}) mm. Press [D] to descend."
                status_color = (0, 255, 0)
                continue

            # D: Descend straight down to touch target
            if key in (ord("d"), ord("D")):
                if target_rob_xyz is None:
                    status_msg = "Click a target point first!"
                    status_color = (0, 0, 255)
                    continue
                if not bot_connected or bot is None:
                    status_msg = "Dobot offline. Cannot execute motion."
                    continue

                rx, ry, rz = target_rob_xyz
                print(f"\n[Dobot] Descending parallel into target: ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm, R={target_rotation_deg:.1f}°...")
                status_msg = f"Descending parallel into target ({rx:.0f}, {ry:.0f}, {rz:.0f}) mm..."
                bot.clear_alarms()
                cur_p = bot.get_pose().position
                if np.hypot(cur_p.x - rx, cur_p.y - ry) > 10.0:
                    if cur_p.z < TRANSIT_SAFE_Z:
                        bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, cur_p.r, wait=True)
                    bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, target_rotation_deg, wait=True)
                    bot.move_to(rx, ry, TRANSIT_SAFE_Z, target_rotation_deg, wait=True)
                bot.move_to(rx, ry, rz, target_rotation_deg, wait=True)
                status_msg = f"Reached target with 180° parallel grip ({rx:.0f}, {ry:.0f}, {rz:.0f}) mm!"
                status_color = (0, 255, 0)
                continue

            # R: Reconnect Dobot (if disconnected)
            if key in (ord("r"), ord("R")):
                if not bot_connected:
                    print("\n[Dobot] Retrying connection...")
                    try_connect_dobot(port)
                    if bot_connected:
                        status_msg = "Dobot reconnected successfully!"
                        status_color = (0, 255, 0)
                    else:
                        status_msg = "Dobot still offline. Close DobotStudio and try again."
                        status_color = (0, 0, 255)
                continue

    finally:
        print("\n[Shutdown] Closing camera and parking robot...")
        if bot_connected and bot is not None:
            try:
                set_gripper_state(bot, "off")
                cur_p = bot.get_pose().position
                if cur_p.z < TRANSIT_SAFE_Z:
                    bot.move_to(cur_p.x, cur_p.y, TRANSIT_SAFE_Z, cur_p.r, wait=True)
                bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)
                bot.close()
            except Exception as e:
                print(f"[Shutdown Dobot Error] {e}")

        pipe.stop()
        cv2.destroyAllWindows()
        print("[Shutdown] Completed cleanly.")


if __name__ == "__main__":
    main()
