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
DEFAULT_SPINE_ROTATION_DEG = 89.70  # Verified ground-truth from DobotStudio for target book (J1=89.70°, J4=0.0°)
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


def compute_parallel_rotation(x, y):
    """
    Computes gripper rotation R parallel to the book spine.
    When Dobot points at the shelf target (x, y) with base azimuth
    J1 = arctan2(y, x), keeping wrist J4 = 0° aligns gripper jaws parallel to
    the upright spine: R = J1 = arctan2(y, x), verified in DobotStudio (R=82.2° at X=35.5, Y=258.4).
    """
    az = float(np.degrees(np.arctan2(y, x)))
    return round(az, 1)


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

    # 3. Transit overhead to target (rx, ry) strictly at Maximum Height
    notify(3, 8, f"Transiting overhead to ({rx:.1f}, {ry:.1f}, {z_max_shelf:.1f}) mm at maximum height...")
    bot.move_to(rx, ry, z_max_shelf, target_r_deg, wait=True)
    time.sleep(0.3)

    # 4. Strictly Vertical Descent straight down into book spine (NO horizontal drift)
    notify(4, 8, f"Descending vertically straight down onto book spine at Z={rz:.1f} mm...")
    bot.move_to(rx, ry, rz, target_r_deg, wait=True)
    time.sleep(0.4)

    # 5. Clamp pneumatic jaws firmly onto spine (full pressure dwell for tight grip)
    notify(5, 8, "Clamping pneumatic jaws firmly onto spine (full pressure dwell)...")
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

    # 1. Load Calibration Transformation (R, t)
    if not c2r.is_calibrated():
        R_current = np.array([[ 0.394764,  0.0, -0.908681],
                              [ 0.852790,  0.0,  0.064282],
                              [ 0.0,      -0.980158, -0.020429]], dtype=float)
        t_current = np.array([295.236, 345.090, 118.218], dtype=float)
        c2r.save(R_current, t_current)
    else:
        R_current, t_current = c2r.load()
        R_current = np.asarray(R_current, float)
        t_current = np.asarray(t_current, float)

    print(f"\n[Calib] Loaded R:\n{np.round(R_current, 3)}")
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
    target_rotation_deg = DEFAULT_SPINE_ROTATION_DEG
    auto_parallel_mode = True
    current_drop_pose = list(DEFAULT_DROP_POSE)

    status_msg = f"Ready. Grasp R={target_rotation_deg:.1f}° [PARALLEL TO SPINE]. Click target, then press P to Retrieve & Drop."
    status_color = (0, 255, 0)
    teach_pairs_cam = []
    teach_pairs_rob = []

    print("\n" + "=" * 68)
    print("     CAMERA 3D -> EQUATION -> DOBOT RETRIEVE & DROP SYSTEM")
    print("=" * 68)
    print("  Click Target : Select book spine coordinate on camera feed")
    print("  P            : FULL RETRIEVE & DROP SEQUENCE (Max Ht J2 -> Grip -> Extract -> Drop)")
    print("  G            : Move to HOVER overhead (Max height using J2, jaws parallel)")
    print("  D            : Move to DESCEND straight down into book spine")
    print("  B            : TEST MOVE TO DROP ZONE (Previews where book will drop)")
    print("  S            : SET DROP ZONE to current physical robot position")
    print("  R            : FLIP GRIPPER 90° (Toggle Parallel <-> Perpendicular)")
    print("  [ / ]        : TUNE GRASP ANGLE +/- 1°")
    print("  SPACE        : Toggle Gripper (Open <-> Clamp)")
    print("  X            : Relax Air Pump (Off)")
    print("  H            : Return to Home Standby Pose")
    print("  T            : TEACH / ALIGN Camera Calibration with robot tip")
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
                        target_rotation_deg = compute_parallel_rotation(target_rob_xyz[0], target_rob_xyz[1])

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

            keys_text = "P:Retrieve & Drop | G:Hover | D:Descend | B:DropZone | S:SetDrop | R:Flip 90° | H:Home"
            cv2.putText(display_img, keys_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            rob_state_text = f"Dobot: X={cur_pose.x:.1f} Y={cur_pose.y:.1f} Z={cur_pose.z:.1f}mm R={cur_pose.r:.1f}° | Grip: {gripper_state.upper()}"
            cv2.putText(display_img, rob_state_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

            par_r = compute_parallel_rotation(target_rob_xyz[0], target_rob_xyz[1]) if target_rob_xyz is not None else DEFAULT_SPINE_ROTATION_DEG
            is_parallel = abs(target_rotation_deg - par_r) < 10.0
            is_perpendicular = abs(abs(target_rotation_deg - par_r) - 90.0) < 10.0
            rot_mode = "PARALLEL TO SPINE" if is_parallel else ("PERPENDICULAR" if is_perpendicular else "CUSTOM")
            rot_color = (0, 255, 0) if is_parallel else (0, 200, 255)

            rot_text = f"Grasp R: {target_rotation_deg:+.1f}° [{rot_mode}] | Drop Zone: [{current_drop_pose[0]:.0f}, {current_drop_pose[1]:.0f}, {current_drop_pose[2]:.0f}] mm"
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

            # G: Hover overhead of target at maximum height
            if key in (ord("g"), ord("G")):
                if target_rob_xyz is None:
                    status_msg = "Please click a target on the camera feed first!"
                    status_color = (0, 0, 255)
                    continue

                rx, ry, rz = target_rob_xyz
                z_max_shelf = compute_max_height_at(rx, ry, fallback=TRANSIT_MAX_HEIGHT_Z)

                print(f"\n[Dobot] Hovering overhead at maximum height: Target=({rx:.1f}, {ry:.1f}), Z={z_max_shelf:.1f}, R={target_rotation_deg:.1f}°...")
                status_msg = f"Hovering at ({rx:.0f}, {ry:.0f}, {z_max_shelf:.0f}) mm, R={target_rotation_deg:.1f}°..."
                bot.clear_alarms()
                cur_p = bot.get_pose().position
                # 1. Elevate to maximum height first
                if cur_p.z < TRANSIT_MAX_HEIGHT_Z:
                    bot.move_to(cur_p.x, cur_p.y, TRANSIT_MAX_HEIGHT_Z, cur_p.r, wait=True)
                # 2. Rotate gripper at maximum height
                bot.move_to(cur_p.x, cur_p.y, TRANSIT_MAX_HEIGHT_Z, target_rotation_deg, wait=True)
                # 3. Move overhead to target
                bot.move_to(rx, ry, z_max_shelf, target_rotation_deg, wait=True)
                status_msg = f"Overhead at ({rx:.0f}, {ry:.0f}, {z_max_shelf:.0f}) mm. Press D to Descend, P to Pick & Drop."
                status_color = (0, 255, 0)
                continue

            # D: Descend strictly vertically to target
            if key in (ord("d"), ord("D")):
                if target_rob_xyz is None:
                    status_msg = "Please click a target on the camera feed first!"
                    status_color = (0, 0, 255)
                    continue

                rx, ry, rz = target_rob_xyz
                z_max_shelf = compute_max_height_at(rx, ry, fallback=TRANSIT_MAX_HEIGHT_Z)

                print(f"\n[Dobot] Descending to target: ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm, R={target_rotation_deg:.1f}°...")
                status_msg = f"Descending to target ({rx:.0f}, {ry:.0f}, {rz:.0f}) mm..."
                bot.clear_alarms()
                cur_p = bot.get_pose().position

                # If not already directly overhead, position overhead at maximum height first
                if np.hypot(cur_p.x - rx, cur_p.y - ry) > 8.0:
                    if cur_p.z < TRANSIT_MAX_HEIGHT_Z:
                        bot.move_to(cur_p.x, cur_p.y, TRANSIT_MAX_HEIGHT_Z, cur_p.r, wait=True)
                    bot.move_to(cur_p.x, cur_p.y, TRANSIT_MAX_HEIGHT_Z, target_rotation_deg, wait=True)
                    bot.move_to(rx, ry, z_max_shelf, target_rotation_deg, wait=True)

                # Strictly vertical descent
                bot.move_to(rx, ry, rz, target_rotation_deg, wait=True)
                status_msg = f"Reached spine ({rx:.0f}, {ry:.0f}, {rz:.0f}) mm, R={target_rotation_deg:.1f}°. Press SPACE to clamp."
                status_color = (0, 255, 0)
                continue

            # R: Toggle 90 deg rotation (Parallel <-> Perpendicular)
            if key in (ord("r"), ord("R")):
                auto_parallel_mode = False
                base_r = compute_parallel_rotation(target_rob_xyz[0], target_rob_xyz[1]) if target_rob_xyz is not None else DEFAULT_SPINE_ROTATION_DEG
                if abs(target_rotation_deg - base_r) < 10.0:
                    target_rotation_deg = round(base_r - 90.0, 1)
                else:
                    target_rotation_deg = base_r
                print(f"[Rotation] Grasp R toggled to: {target_rotation_deg:.1f}°")
                status_msg = f"Grasp R toggled to {target_rotation_deg:.1f}° ({'PARALLEL TO SPINE' if abs(target_rotation_deg - base_r) < 10.0 else 'PERPENDICULAR'})"
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
                    c2r.save(R_current, t_current)
                    print(f"[Teach] Point {len(teach_pairs_cam)} aligned! Translation updated to t={np.round(t_current, 2)} mm")
                    status_msg = f"Point {len(teach_pairs_cam)} aligned! t updated to [{t_current[0]:.0f}, {t_current[1]:.0f}, {t_current[2]:.0f}]"
                else:
                    R_opt, t_opt = c2r.fit_rigid(teach_pairs_cam, teach_pairs_rob)
                    errs = c2r.residuals(R_opt, t_opt, teach_pairs_cam, teach_pairs_rob)
                    rmse = float(np.sqrt(np.mean(errs ** 2)))
                    R_current, t_current = R_opt, t_opt
                    c2r.save(R_current, t_current)
                    print(f"[Teach] Full SVD fit updated on {len(teach_pairs_cam)} points! RMSE = {rmse:.2f} mm")
                    status_msg = f"SVD Fit Updated ({len(teach_pairs_cam)} pts)! RMSE = {rmse:.1f} mm"

                status_color = (0, 255, 0)
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
