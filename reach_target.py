"""
reach_target.py
--------------------------------------------------------------------------------
Direct 3D Camera -> Equation -> Dobot Magician Book Retrieval & Drop System.

Equations applied:
  1. Deprojection (Pixel u,v + Depth -> Camera 3D Xc, Yc, Zc in mm):
       Xc = (u - cx) * depth / fx
       Yc = (v - cy) * depth / fy
       Zc = depth

  2. Extrinsics (Camera 3D -> Dobot Base 3D Xr, Yr, Zr in mm):
       [Xr, Yr, Zr]^T = R @ [Xc, Yc, Zc]^T + t

  3. Grasp Alignment:
       R_grasp = -2.31 deg (exact verified parallel-to-spine angle)
       Interactive adjustments: R key (flip 90 deg), [ / ] key (+/- 5 deg)

  4. Full Retrieve & Drop Pipeline:
       - Open gripper & transit overhead (Hover)
       - Descend onto book spine with jaws parallel
       - Clamp pneumatic jaws firmly (dwell 0.8s)
       - Smooth vertical lift to extract book clear of shelf (+40mm)
       - High-clearance transfer across to Deposit Station (Drop Zone)
       - Lower into deposit tray, open jaws to release, relax air pump
       - Retract arm safely back to standby Home pose
--------------------------------------------------------------------------------
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
from pydobotplus import Dobot
from pydobotplus.message import Message
import cam2robot as c2r
import dobot_kin as K

HOME_POSE = (200.0, 0.0, 160.0, 0.0)
DEFAULT_DROP_POSE = [200.0, 0.0, 165.0, 0.0]  # Front drop zone at maximum height (X, Y, Z, R)
DEFAULT_PORT = "COM7"
W, H, FPS = 640, 480, 30
DEFAULT_SPINE_ROTATION_DEG = 88.10  # Verified ground-truth from DobotStudio for target book (J1=88.10°, J4=0.0°, R=88.10°)
DEFAULT_ROTATION_BY_MODE = {
    "spine": 88.1,
    "scattered": 94.2,
}
TRANSIT_MAX_HEIGHT_Z = 165.0       # Maximum safe ceiling height reachable via J2


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
    """
    Direct pneumatic gripper control:
      'open'  -> Valve RELEASE, Pump ON
      'close' -> Valve GRIP, Pump ON
      'off'   -> Pump OFF (relax pressure)
    """
    msg = Message()
    msg.id = 63
    msg.ctrl = 0x03
    if state == "close":
        msg.params = bytearray([0x01, 0x01])
    elif state == "open":
        msg.params = bytearray([0x01, 0x00])
    else:
        msg.params = bytearray([0x00, 0x00])
    return bot._send_command(msg)


def deproject_pixel_to_cam_mm(u, v, depth_m, intr):
    """
    Equation 1: Intrinsic pinhole projection to Camera 3D in mm
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth
    """
    pt = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], depth_m)
    return np.array([pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0])


def cam_mm_to_robot_mm(cam_xyz_mm, R, t):
    """
    Equation 2: Extrinsic Rigid Transformation to Robot Base 3D in mm
    P_robot = R @ P_cam + t
    """
    return R @ cam_xyz_mm + t


def compute_parallel_rotation(x, y, mode="spine", spine_orientation="vertical"):
    """
    Computes gripper rotation R based on arrangement mode:
      - 'spine':
          * 'vertical': Gripper aligns parallel along book spine depth/height.
          * 'perpendicular': 90° rotated across the book spine.
      - 'scattered':
          * Radial approach across desk: R = J1 = arctan2(y, x).
    """
    if mode in ("scattered", "table", "laying"):
        j1 = float(np.degrees(np.arctan2(y, x)))
        return round(j1, 1)

    # Spine Mode
    base_r = float(c2r.get_spine_parallel_rotation(x, y))
    if spine_orientation == "perpendicular":
        r_perp = base_r - 90.0
        return round((r_perp + 180.0) % 360.0 - 180.0, 1)
    return round(base_r, 1)


def compute_max_height_at(x, y, fallback=149.0):
    """
    Calculates the maximum safe height reachable at (x, y) using J2 elevation.
    """
    try:
        eq = K.stage_equations(x, y, 130.0)
        if eq.get("ok") and "z_A" in eq:
            return float(min(eq["z_A"], 165.0))
    except Exception:
        pass
    r = float(np.hypot(x, y))
    zm = K.z_max_at(r)
    return float(min(zm, 165.0)) if zm is not None else fallback


def execute_full_retrieve_and_drop(bot, rx, ry, rz, target_r_deg, drop_pose, update_ui=None):
    """
    Executes complete 8-stage autonomous retrieval & drop motion:
      1. Ascend to Maximum Height using J2 (Clearance Elevation) & open gripper
      2. Pre-Rotate Gripper to target angle at Maximum Height
      3. Transit horizontally overhead to (rx, ry) strictly at Maximum Height
      4. Strictly vertical descent straight down into book spine
      5. Clamp pneumatic jaws firmly on spine (dwell 0.8s)
      6. Strictly vertical extraction: Lift straight up to Maximum Height using J2
      7. Transfer to Drop Zone at Maximum Height & Drop book directly at Maximum Height
      8. Retract arm cleanly to Home standby
    """
    dx, dy, dz, dr = drop_pose
    z_max_shelf = compute_max_height_at(rx, ry, fallback=149.0)
    transit_z = float(min(165.0, z_max_shelf))

    bot.clear_alarms()
    cur = bot.get_pose().position

    def notify(step_num, total, text):
        msg = f"[{step_num}/{total}] {text}"
        print(f"\n{msg}", flush=True)
        if update_ui:
            update_ui(msg)

    # 1. First: Elevate to Maximum Height using J2 before ANY horizontal movement
    notify(1, 8, f"Ascending to Maximum Height Z={transit_z:.0f} mm using J2...")
    set_gripper_state(bot, "open")
    time.sleep(0.2)
    bot.move_to(cur.x, cur.y, transit_z, cur.r, wait=True)

    # 2. Pre-Rotate Gripper to target R at Maximum Height (Overhead in the clear)
    notify(2, 8, f"Pre-rotating gripper to R={target_r_deg:.1f}° at maximum height...")
    bot.move_to(cur.x, cur.y, transit_z, target_r_deg, wait=True)

    try:
        import pydobotplus.dobotplus as dp
        movl_mode = dp.MODE_PTP.MOVL_XYZ
    except Exception:
        movl_mode = None

    target_pick_z = float(max(rz, 2.0))

    # 3. Transit overhead to target (rx, ry) strictly at Maximum Height ("go there")
    notify(3, 8, f"Transiting overhead to ({rx:.1f}, {ry:.1f}, {z_max_shelf:.1f}) mm at maximum height...")
    bot.move_to(rx, ry, z_max_shelf, target_r_deg, wait=True)
    time.sleep(0.3)

    # 4. Strictly Vertical Descent straight down into book pick point at target_pick_z ("and go down")
    notify(4, 8, f"Descending vertically straight down onto book at Z={target_pick_z:.1f} mm...")
    bot.move_to(rx, ry, target_pick_z, target_r_deg, wait=True)
    time.sleep(0.4)

    # 5. Clamp pneumatic jaws firmly onto target (full pressure dwell for tight grip)
    notify(5, 8, "Clamping pneumatic jaws firmly onto target (full pressure dwell)...")
    set_gripper_state(bot, "close")
    time.sleep(1.2)

    # 6. Strictly Vertical Extraction: Lift book straight up using J2 to Maximum Height
    notify(6, 8, f"Extracting book straight up to Maximum Height Z={transit_z:.0f} mm using J2...")
    bot.move_to(rx, ry, transit_z, target_r_deg, wait=True)
    time.sleep(0.3)

    # 7. Transfer to Drop Zone at Maximum Height & Drop at Maximum Height
    notify(7, 8, f"Transferring to Drop Zone ({dx:.1f}, {dy:.1f}) at Maximum Height Z={transit_z:.0f} mm...")
    bot.move_to(dx, dy, transit_z, dr, wait=True)
    time.sleep(0.3)

    # Drop book directly at Maximum Height
    print(f"      Dropping book directly at Maximum Height Z={transit_z:.0f} mm...", flush=True)
    set_gripper_state(bot, "open")
    time.sleep(0.6)
    set_gripper_state(bot, "off")

    # 8. Retract arm back to Home Standby at Maximum Height
    notify(8, 8, f"Retracting arm back to Home Standby ({HOME_POSE[0]:.0f}, {HOME_POSE[1]:.0f}, {HOME_POSE[2]:.0f}) mm...")
    bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)

    print("\n[SUCCESS] Book successfully retrieved from shelf and dropped at maximum height!\n", flush=True)
    if update_ui:
        update_ui("SUCCESS: Book retrieved and dropped at maximum height.")


# Interactive Mouse State
selected_pixel = None
auto_parallel_mode = True


def on_mouse(event, x, y, flags, param):
    global selected_pixel, auto_parallel_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)
        auto_parallel_mode = True


def main():
    global selected_pixel, auto_parallel_mode

    import argparse
    parser = argparse.ArgumentParser(description="Direct 3D Camera -> Dobot Retrieval")
    parser.add_argument(
        "--mode",
        default="spine",
        choices=["spine", "scattered"],
        help="Arrangement mode: spine (default) or scattered",
    )
    args, _ = parser.parse_known_args()
    active_mode = args.mode
    spine_orientation = "vertical"  # In spine mode: 'vertical' (default) or 'perpendicular'

    # 1. Load Calibration Transformation (R, t) for active mode
    if not c2r.is_calibrated():
        R_current = np.array([[ 0.394764,  0.0, -0.908681],
                              [ 0.852790,  0.0,  0.064282],
                              [ 0.0,      -0.980158, -0.020429]], dtype=float)
        t_current = np.array([326.131, 340.413, 119.636], dtype=float)
        c2r.save(R_current, t_current)
    else:
        R_current, t_current = c2r.load(mode=active_mode)
        R_current = np.asarray(R_current, float)
        t_current = np.asarray(t_current, float)

    def save_active_calibration(R, t):
        if active_mode == "scattered":
            c2r.save_scattered(R, t)
        else:
            c2r.save_spine(R, t)

    print(f"\n[Calib] Active Mode: {active_mode.upper()}")
    print(f"[Calib] Loaded R:\n{np.round(R_current, 3)}")
    print(f"[Calib] Loaded t: {np.round(t_current, 2)} mm")

    # 2. Initialize RealSense Camera
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    try:
        prof = pipe.start(cfg)
    except Exception as e:
        print(f"[Camera Error] Failed to start RealSense camera: {e}")
        return

    align = rs.align(rs.stream.color)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # 3. Initialize Dobot Magician
    target_port = auto_detect_dobot_port()
    try:
        bot = Dobot(port=target_port)
    except Exception as e:
        pipe.stop()
        print(f"\n[Dobot Error] Could not connect on {target_port}: {e}")
        print("Please ensure DobotStudio is disconnected and cable is plugged in.")
        return

    print(f"[Dobot] Connected on {target_port}!")
    bot.clear_alarms()
    time.sleep(0.2)
    set_gripper_state(bot, "off")
    gripper_state = "off"

    # Move to Home Pose
    print(f"[Dobot] Moving to Home Pose: {HOME_POSE}...")
    bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)

    # 4. State & Controls
    cv2.namedWindow("Camera -> Robot 3D Reach", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Camera -> Robot 3D Reach", on_mouse)

    target_cam_xyz = None
    target_rob_xyz = None
    last_depth_mm = 0.0
    target_rotation_deg = DEFAULT_SPINE_ROTATION_DEG if active_mode == "spine" else DEFAULT_ROTATION_BY_MODE.get("scattered", 94.2)
    auto_parallel_mode = True
    current_drop_pose = list(DEFAULT_DROP_POSE)

    mode_label = f"SPINE - {spine_orientation.upper()}" if active_mode == "spine" else "SCATTERED"
    status_msg = f"Ready. Mode: {mode_label} | Grasp R={target_rotation_deg:.1f}°. Click target, then press P."
    status_color = (0, 255, 0)
    teach_pairs_cam = []
    teach_pairs_rob = []

    print("\n" + "=" * 68)
    print("     CAMERA 3D -> EQUATION -> DOBOT RETRIEVE & DROP SYSTEM")
    print("=" * 68)
    print(f"  Active Mode  : [{active_mode.upper()}] (Press M to toggle Spine <-> Scattered)")
    print("  Click Target : Select book coordinate on camera feed")
    print("  P            : FULL RETRIEVE & DROP SEQUENCE (Autonomous grasp & deposit)")
    print("  M            : SWITCH MODE (Spine <-> Scattered)")
    print("  R            : TOGGLE SPINE GRASP (Vertical <-> Perpendicular)")
    print("  B            : TEST MOVE TO DROP ZONE (Previews where book will drop)")
    print("  S            : SET DROP ZONE to current physical robot position")
    print("  [ / ]        : TUNE GRASP ANGLE +/- 1°")
    print("  [- / +]      : TUNE Z CALIBRATION +/- 1 mm")
    print("  SPACE        : Toggle Gripper (Open <-> Clamp)")
    print("  X            : Relax Air Pump (Off)")
    print("  H            : Return to Home Standby Pose")
    print("  T            : TEACH / ALIGN Camera Calibration with robot tip")
    print("  C            : CALIBRATE from DobotStudio coordinates (Manual X, Y, Z)")
    print("  Q / ESC      : Quit safely")
    print("=" * 68 + "\n")

    def update_ui_status(text):
        nonlocal status_msg
        status_msg = text

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

            cur_pose = bot.get_pose().position

            if selected_pixel:
                u, v = selected_pixel
                v1, v2 = max(0, v - 2), min(H, v + 3)
                u1, u2 = max(0, u - 2), min(W, u + 3)
                patch = depth_data[v1:v2, u1:u2]
                valid = patch[patch > 0]

                if valid.size > 0:
                    d_m = float(np.median(valid))
                    last_depth_mm = d_m * 1000.0

                    target_cam_xyz = deproject_pixel_to_cam_mm(u, v, d_m, intr)
                    target_rob_xyz = cam_mm_to_robot_mm(target_cam_xyz, R_current, t_current)

                    if auto_parallel_mode:
                        target_rotation_deg = compute_parallel_rotation(
                            target_rob_xyz[0], target_rob_xyz[1], mode=active_mode, spine_orientation=spine_orientation
                        )

                    r_reach = float(np.hypot(target_rob_xyz[0], target_rob_xyz[1]))
                    is_reachable = (140.0 <= r_reach <= 320.0) and (-60.0 <= target_rob_xyz[2] <= 165.0)

                    status_msg = f"Target: X={target_rob_xyz[0]:.1f}, Y={target_rob_xyz[1]:.1f}, Z={target_rob_xyz[2]:.1f} mm | R={target_rotation_deg:.1f}° | Press P to Retrieve & Drop!"
                    status_color = (0, 255, 0) if is_reachable else (0, 0, 255)
                else:
                    status_msg = f"No depth return at ({u}, {v}). Click inside object."
                    status_color = (0, 165, 255)
                    target_cam_xyz = None
                    target_rob_xyz = None

            display_img = color_img.copy()

            # Top banner
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (W, 78), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, H - 35), (W, H), (20, 20, 20), -1)
            display_img = cv2.addWeighted(overlay, 0.75, display_img, 0.25, 0)

            keys_text = "P:Pick&Drop | M:Mode(Spine/Scat) | R:Spine(Vert/Perp) | [-/+]:Trim Z | [/]:Trim R | H:Home | C:Calib"
            cv2.putText(display_img, keys_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            rob_state_text = f"Dobot: X={cur_pose.x:.1f} Y={cur_pose.y:.1f} Z={cur_pose.z:.1f}mm R={cur_pose.r:.1f}° | Grip: {gripper_state.upper()}"
            cv2.putText(display_img, rob_state_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

            if active_mode == "spine":
                mode_label = f"SPINE - {spine_orientation.upper()}"
                rot_color = (0, 255, 0) if spine_orientation == "vertical" else (0, 200, 255)
            else:
                mode_label = "SCATTERED"
                rot_color = (0, 255, 200)

            rot_text = f"Mode: [{mode_label}] | Grasp R: {target_rotation_deg:+.1f}° | Drop: [{current_drop_pose[0]:.0f}, {current_drop_pose[1]:.0f}, {current_drop_pose[2]:.0f}] mm"
            cv2.putText(display_img, rot_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.39, rot_color, 1)

            if selected_pixel and target_rob_xyz is not None:
                u, v = selected_pixel
                cv2.drawMarker(display_img, (u, v), status_color, cv2.MARKER_CROSS, 20, 2)
                cv2.circle(display_img, (u, v), 8, status_color, 1)

                t_info1 = f"Cam 3D: [{target_cam_xyz[0]:.1f}, {target_cam_xyz[1]:.1f}, {target_cam_xyz[2]:.1f}] mm (D={last_depth_mm:.0f}mm)"
                t_info2 = f"Robot 3D: X={target_rob_xyz[0]:.1f}, Y={target_rob_xyz[1]:.1f}, Z={target_rob_xyz[2]:.1f} mm | R={target_rotation_deg:.1f}°"

                card_x = min(u + 15, W - 325)
                card_y = max(v - 15, 95)
                cv2.rectangle(display_img, (card_x - 5, card_y - 20), (card_x + 320, card_y + 25), (0, 0, 0), -1)
                cv2.putText(display_img, t_info1, (card_x, card_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                cv2.putText(display_img, t_info2, (card_x, card_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.39, status_color, 1)

            cv2.putText(display_img, status_msg, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

            cv2.imshow("Camera -> Robot 3D Reach", display_img)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            # Home
            if key in (ord("h"), ord("H")):
                print("\n[Dobot] Returning to Home Pose...")
                status_msg = "Returning to Home Pose..."
                bot.clear_alarms()
                bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)
                status_msg = "At Home Standby Pose."
                continue

            # M: Toggle Mode between Spine and Scattered
            if key in (ord("m"), ord("M")):
                active_mode = "scattered" if active_mode == "spine" else "spine"
                auto_parallel_mode = True
                R_current, t_current = c2r.load(mode=active_mode)
                R_current = np.asarray(R_current, float)
                t_current = np.asarray(t_current, float)
                if target_cam_xyz is not None:
                    target_rob_xyz = cam_mm_to_robot_mm(target_cam_xyz, R_current, t_current)
                    target_rotation_deg = compute_parallel_rotation(
                        target_rob_xyz[0], target_rob_xyz[1], mode=active_mode, spine_orientation=spine_orientation
                    )
                else:
                    target_rotation_deg = DEFAULT_SPINE_ROTATION_DEG if active_mode == "spine" else DEFAULT_ROTATION_BY_MODE.get("scattered", 94.2)
                mode_label = f"SPINE - {spine_orientation.upper()}" if active_mode == "spine" else "SCATTERED"
                print(f"\n[Mode] Switched to {active_mode.upper()} mode! Loaded {active_mode} calibration (t={np.round(t_current, 1)} mm).")
                status_msg = f"Mode: {mode_label} | Grasp R: {target_rotation_deg:.1f}°"
                status_color = (0, 255, 255)
                continue

            # P: Full Retrieve & Drop Sequence
            if key in (ord("p"), ord("P")):
                if target_rob_xyz is None:
                    status_msg = "Please click a target on the camera feed first!"
                    status_color = (0, 0, 255)
                    continue

                rx, ry, rz = target_rob_xyz
                execute_full_retrieve_and_drop(
                    bot, rx, ry, rz, target_rotation_deg, current_drop_pose, update_ui=update_ui_status
                )
                gripper_state = "off"
                continue

            # B: Test Move to Drop Zone (At Maximum Height)
            if key in (ord("b"), ord("B")):
                dx, dy, dz, dr = current_drop_pose
                print(f"\n[Drop Zone Preview] Moving to Drop Zone at Max Height: ({dx:.1f}, {dy:.1f}, {TRANSIT_MAX_HEIGHT_Z:.1f}) mm...")
                status_msg = f"Testing Drop Zone ({dx:.0f}, {dy:.0f}, {TRANSIT_MAX_HEIGHT_Z:.0f}) mm at Max Ht..."
                bot.clear_alarms()
                cur_pose = bot.get_pose().position
                if cur_pose.z < TRANSIT_MAX_HEIGHT_Z:
                    bot.move_to(cur_pose.x, cur_pose.y, TRANSIT_MAX_HEIGHT_Z, cur_pose.r, wait=True)
                bot.move_to(dx, dy, TRANSIT_MAX_HEIGHT_Z, dr, wait=True)
                status_msg = f"At Drop Zone ({dx:.0f}, {dy:.0f}, {TRANSIT_MAX_HEIGHT_Z:.0f}) mm at Max Ht. Press H to return Home."
                continue

            # S: Set Drop Zone to Current Robot Position
            if key in (ord("s"), ord("S")):
                cur_p = bot.get_pose().position
                current_drop_pose = [round(cur_p.x, 1), round(cur_p.y, 1), round(cur_p.z, 1), round(cur_p.r, 1)]
                print(f"\n[Drop Zone Set] Drop location updated to: {current_drop_pose}")
                status_msg = f"Drop Zone updated to ({current_drop_pose[0]:.0f}, {current_drop_pose[1]:.0f}, {current_drop_pose[2]:.0f}) mm"
                continue

            # R: In Spine mode, toggle Vertical <-> Perpendicular; in Scattered, toggle 90°
            if key in (ord("r"), ord("R")):
                auto_parallel_mode = False
                if active_mode == "spine":
                    spine_orientation = "perpendicular" if spine_orientation == "vertical" else "vertical"
                    if target_rob_xyz is not None:
                        target_rotation_deg = compute_parallel_rotation(
                            target_rob_xyz[0], target_rob_xyz[1], mode="spine", spine_orientation=spine_orientation
                        )
                    else:
                        base_r = DEFAULT_SPINE_ROTATION_DEG
                        target_rotation_deg = base_r if spine_orientation == "vertical" else round((base_r - 90.0 + 180.0) % 360.0 - 180.0, 1)
                    print(f"[Spine Grasp] Orientation switched to: {spine_orientation.upper()} (R={target_rotation_deg:.1f}°)")
                    status_msg = f"Spine Grasp: {spine_orientation.upper()} ({target_rotation_deg:.1f}°)"
                else:
                    target_rotation_deg = round((target_rotation_deg - 90.0 + 180.0) % 360.0 - 180.0, 1)
                    print(f"[Scattered Grasp] Rotation toggled: R={target_rotation_deg:.1f}°")
                    status_msg = f"Scattered Grasp R: {target_rotation_deg:.1f}°"
                continue

            # [ and ]: Tune rotation by +/- 1 degree
            if key == ord("["):
                auto_parallel_mode = False
                target_rotation_deg = round(target_rotation_deg - 1.0, 1)
                print(f"[Rotation] Grasp R: {target_rotation_deg:.1f}°")
                status_msg = f"Grasp R trimmed: {target_rotation_deg:.1f}°"
                continue

            if key == ord("]"):
                auto_parallel_mode = False
                target_rotation_deg = round(target_rotation_deg + 1.0, 1)
                print(f"[Rotation] Grasp R: {target_rotation_deg:.1f}°")
                status_msg = f"Grasp R trimmed: {target_rotation_deg:.1f}°"
                continue

            # - and + (=): Tune Z calibration height live by +/- 1 mm
            if key in (ord("-"), ord("_")):
                t_current[2] -= 1.0
                save_active_calibration(R_current, t_current)
                print(f"[Z-Trim] [{active_mode.upper()}] Calibration Z lowered by 1 mm -> t[Z]={t_current[2]:.2f} mm")
                status_msg = f"[{active_mode.upper()}] Z trimmed -1mm: t[Z]={t_current[2]:.1f}mm"
                continue

            if key in (ord("="), ord("+")):
                t_current[2] += 1.0
                save_active_calibration(R_current, t_current)
                print(f"[Z-Trim] [{active_mode.upper()}] Calibration Z raised by 1 mm -> t[Z]={t_current[2]:.2f} mm")
                status_msg = f"[{active_mode.upper()}] Z trimmed +1mm: t[Z]={t_current[2]:.1f}mm"
                continue

            # Spacebar: Toggle Gripper
            if key == 32:
                new_st = "close" if gripper_state != "close" else "open"
                print(f"[Gripper] Setting -> {new_st.upper()}...")
                set_gripper_state(bot, new_st)
                gripper_state = new_st
                continue

            # X: Gripper pump OFF
            if key in (ord("x"), ord("X")):
                print("[Gripper] Pump turned OFF...")
                set_gripper_state(bot, "off")
                gripper_state = "off"
                continue

            # T: Teach / Align
            if key in (ord("t"), ord("T")):
                if target_cam_xyz is None:
                    status_msg = "Click the target point on camera first, then press T to align!"
                    status_color = (0, 0, 255)
                    continue

                cur_p = bot.get_pose().position
                real_rob_pt = np.array([cur_p.x, cur_p.y, cur_p.z], dtype=float)

                print(f"\n[Teach] Current Robot Tip: X={real_rob_pt[0]:.2f}, Y={real_rob_pt[1]:.2f}, Z={real_rob_pt[2]:.2f}")
                print(f"[Teach] Target Cam 3D:    X={target_cam_xyz[0]:.2f}, Y={target_cam_xyz[1]:.2f}, Z={target_cam_xyz[2]:.2f}")

                teach_pairs_cam.append(target_cam_xyz.copy())
                teach_pairs_rob.append(real_rob_pt.copy())

                if len(teach_pairs_cam) < 3:
                    t_current = real_rob_pt - (R_current @ target_cam_xyz)
                    save_active_calibration(R_current, t_current)
                    print(f"[Teach] Point {len(teach_pairs_cam)} aligned! Translation updated to t={np.round(t_current, 2)} mm")
                    status_msg = f"Point {len(teach_pairs_cam)} aligned! t updated to [{t_current[0]:.0f}, {t_current[1]:.0f}, {t_current[2]:.0f}]"
                else:
                    R_opt, t_opt = c2r.fit_rigid(teach_pairs_cam, teach_pairs_rob)
                    errs = c2r.residuals(R_opt, t_opt, teach_pairs_cam, teach_pairs_rob)
                    rmse = float(np.sqrt(np.mean(errs ** 2)))
                    R_current, t_current = R_opt, t_opt
                    save_active_calibration(R_current, t_current)
                    print(f"[Teach] Full SVD fit updated on {len(teach_pairs_cam)} points! RMSE = {rmse:.2f} mm")
                    status_msg = f"SVD Fit Updated ({len(teach_pairs_cam)} pts)! RMSE = {rmse:.1f} mm"

                status_color = (0, 255, 0)
                continue

            # C: Calibrate translation from manual DobotStudio coordinates
            if key in (ord("c"), ord("C")):
                if target_cam_xyz is None:
                    status_msg = "Click the target point on camera first, then press C to calibrate!"
                    status_color = (0, 0, 255)
                    continue

                print("\n" + "=" * 65)
                print(f"[Calibrate] Selected Camera 3D: [{target_cam_xyz[0]:.1f}, {target_cam_xyz[1]:.1f}, {target_cam_xyz[2]:.1f}] mm")
                print("Enter Dobot coordinates from DobotStudio Operation Panel:")
                print("Format: X Y Z   (e.g., '18.9 283.2 126.3' or leave blank to cancel)")
                print("=" * 65)
                try:
                    coords_input = input("Dobot [X Y Z]: ").strip()
                    if coords_input:
                        parts = [float(val) for val in coords_input.replace(",", " ").split() if val.strip()]
                        if len(parts) >= 3:
                            p_rob_manual = np.array(parts[:3], dtype=float)
                            t_current = p_rob_manual - (R_current @ target_cam_xyz)
                            save_active_calibration(R_current, t_current)
                            print(f"[Calibrate] SUCCESS! Translation updated for [{active_mode.upper()}] to t = {np.round(t_current, 2)} mm")
                            print(f"[Calibrate] Recalibrated robot target: {np.round(R_current @ target_cam_xyz + t_current, 2)} mm")
                            status_msg = f"Calibrated [{active_mode.upper()}]! Robot Target: X={p_rob_manual[0]:.1f}, Y={p_rob_manual[1]:.1f}, Z={p_rob_manual[2]:.1f}"
                            status_color = (0, 255, 0)
                        else:
                            print("[Calibrate] Invalid input. Expected 3 numbers: X Y Z")
                    else:
                        print("[Calibrate] Cancelled.")
                except Exception as ex:
                    print(f"[Calibrate Error]: {ex}")
                continue

    finally:
        print("\n[Shutdown] Parking Dobot Magician and stopping camera...")
        try:
            set_gripper_state(bot, "off")
            bot.clear_alarms()
            bot.move_to(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], wait=True)
            bot.close()
        except Exception as e:
            print("[Shutdown Dobot error]:", e)

        pipe.stop()
        cv2.destroyAllWindows()
        print("[Shutdown] Completed.")


if __name__ == "__main__":
    main()
