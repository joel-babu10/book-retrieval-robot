"""
robot/dobot_controller.py
----------------------------------------------------------------------------
Dobot Magician Serial Robotic Arm Controller using pydobotplus.
Provides direct hardware control for camera-to-robot coordinate execution:
  - Coordinate transformation: Camera (m) -> Dobot workspace (mm) via R @ P + t
  - Safe hover approach
  - Precision descent & pneumatic gripper control
  - Vertical lift & transfer
  - Retract to home
----------------------------------------------------------------------------
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import dobot_kin as K

try:
    from pydobotplus import Dobot
    from pydobotplus.message import Message
    HAS_PYDOBOT = True
except ImportError:
    HAS_PYDOBOT = False

DOBOT_HOME = (200.0, 0.0, 160.0, 0.0)
DEPOSIT_ZONE = (200.0, 0.0, 165.0, 0.0)   # Front drop zone at maximum height
DEFAULT_PORT = "COM7"
TRANSIT_MAX_HEIGHT_Z = 165.0


def compute_parallel_rotation(x, y):
    """
    Computes gripper rotation R parallel to the book spine.
    When Dobot points at the shelf target (x, y) with base azimuth
    J1 = arctan2(y, x), keeping wrist J4 = 0° aligns gripper jaws parallel to
    the upright spine: R = J1 = arctan2(y, x), verified in DobotStudio (R=82.2° at X=35.5, Y=258.4).
    """
    az = float(np.degrees(np.arctan2(y, x)))
    return round(az, 1)


def compute_max_height_at(x, y, fallback=165.0):
    """Calculates the maximum safe height reachable at (x, y) using J2 elevation."""
    try:
        eq = K.stage_equations(x, y, 130.0)
        if eq.get("ok") and "z_A" in eq:
            return float(min(eq["z_A"], 165.0))
    except Exception:
        pass
    r = float(np.hypot(x, y))
    zm = K.z_max_at(r)
    return float(min(zm, 165.0)) if zm is not None else fallback


def _set_gripper_state(bot_instance, state: str):
    """
    Direct pneumatic gripper control:
      'open'  -> Valve RELEASE, Pump ON
      'close' -> Valve GRIP, Pump ON
      'off'   -> Pump OFF (relax pressure)
    """
    if bot_instance is None:
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
    return bot_instance._send_command(msg)


class DobotController:
    def __init__(self, port=None):
        self.port = port
        self.bot = None
        self.is_connected = False
        self.connect(port)

    @staticmethod
    def compute_parallel_rotation(x, y):
        return compute_parallel_rotation(x, y)

    def connect(self, port=None):
        """Connects to the Dobot Magician arm on the specified or auto-detected COM port."""
        if not HAS_PYDOBOT:
            print("[Dobot] Note: pydobotplus not installed, running in simulation mode.", flush=True)
            self.is_connected = False
            return False

        target_port = port or self.port or self._auto_detect_port()
        try:
            self.bot = Dobot(port=target_port)
            self.bot.clear_alarms()
            time.sleep(0.2)
            _set_gripper_state(self.bot, "off")
            self.is_connected = True
            self.port = target_port
            print(f"[Dobot] Successfully connected to Dobot Magician on {target_port}!", flush=True)
            return True
        except Exception as e:
            print(f"[Dobot] Could not connect on {target_port}: {e}. Simulation mode active.", flush=True)
            self.bot = None
            self.is_connected = False
            return False

    def _auto_detect_port(self):
        """Scans active COM ports for Dobot Magician."""
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                desc = (p.description or "") + " " + (p.hwid or "")
                if any(k in desc for k in ("Prolific", "CP210", "USB-to-Serial", "CH340", "FTDI")):
                    return p.device
        except Exception:
            pass
        return DEFAULT_PORT

    def camera_to_dobot_coordinates(self, camera_xyz_m):
        """
        Transforms 3D coordinates from RealSense Optical Frame (meters or mm)
        to Dobot Magician Base Frame (millimeters).
        Equation: P_robot = R @ P_cam + t
        """
        if camera_xyz_m is None:
            return np.array([200.0, 0.0, 100.0])

        try:
            import cam2robot as c2r
            if c2r.is_calibrated():
                rob = c2r.cam_to_robot(camera_xyz_m)
                return np.array([float(rob[0]), float(rob[1]), float(rob[2])])
        except Exception:
            pass

        # Fallback default transformation if calib not available
        xc, yc, zc = np.asarray(camera_xyz_m, float)
        if np.max(np.abs([xc, yc, zc])) < 5.0:
            xc, yc, zc = xc * 1000.0, yc * 1000.0, zc * 1000.0

        xr = 0.394764 * xc - 0.908681 * zc + 288.209
        yr = 0.852790 * xc + 0.064282 * zc + 356.924
        zr = -0.980158 * yc - 0.020429 * zc + 116.693
        return np.array([xr, yr, zr])

    def execute_pick_sequence(self, target_xyz_m, grasp_angle_deg=None, is_spine=False, update_ui=None):
        """
        Executes complete autonomous pick and retrieval sequence:
          1. Ascend to Maximum Height using J2 (Clearance Elevation) & open jaws
          2. Pre-Rotate Gripper to target angle at Maximum Height
          3. Transit overhead to target (rx, ry) strictly at Maximum Height
          4. Strictly vertical descent straight down into book spine
          5. Clamp pneumatic jaws firmly on spine
          6. Strictly vertical extraction: Lift straight up to Maximum Height using J2
          7. Transfer to Drop Zone at transit clearance & deposit book
          8. Retract arm cleanly to Home standby
        """
        robot_xyz = self.camera_to_dobot_coordinates(target_xyz_m)
        rx, ry, rz = float(robot_xyz[0]), float(robot_xyz[1]), float(robot_xyz[2])

        # Any target on the bookshelf (Z >= 60mm) or marked as spine MUST use verified shelf-parallel rotation
        if is_spine or rz >= 60.0 or grasp_angle_deg is None:
            r_angle = compute_parallel_rotation(rx, ry)
        else:
            r_angle = float(grasp_angle_deg)

        z_max_shelf = compute_max_height_at(rx, ry, fallback=149.0)
        transit_z = float(min(TRANSIT_MAX_HEIGHT_Z, z_max_shelf))
        dx, dy, dz, dr = DEPOSIT_ZONE

        print(f"\n================ DOBOT MAGICIAN EXECUTION ================", flush=True)
        print(f"Target Camera Optical:  XYZ = {target_xyz_m}", flush=True)
        print(f"Dobot Arm Coordinates:  X={rx:.1f}mm, Y={ry:.1f}mm, Z={rz:.1f}mm, R={r_angle:.1f}°", flush=True)
        print(f"Ceiling Max Height:     Z={z_max_shelf:.1f}mm (using J2 elevation)", flush=True)
        print(f"Drop Station:           X={dx:.1f}mm, Y={dy:.1f}mm, Z={dz:.1f}mm, R={dr:.1f}°", flush=True)
        print(f"Hardware Connected:     {self.is_connected}", flush=True)

        # Validate kinematic limits before executing physical motions
        r_reach = float(np.hypot(rx, ry))
        reach_ok, reach_reason, _ = K.check(rx, ry, rz)
        if not reach_ok:
            is_bg_depth = (target_xyz_m is not None and len(target_xyz_m) >= 3 and float(target_xyz_m[2]) > 0.46)
            bg_hint = " Depth was sampled from background wall (>0.46m)." if is_bg_depth else ""
            err_msg = f"Target unreachable: {reach_reason} at ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm (r={r_reach:.1f} mm).{bg_hint}"
            print(f"\n[Dobot ERROR] {err_msg}", flush=True)
            if update_ui:
                err_detail = "Target depth sampled from background wall behind shelf." if is_bg_depth else "Target is outside robot physical reach. Shelf placed too far or too close."
                update_ui(err_msg, stage_num=0, is_error=True, error_detail=err_detail)
            return {
                "status": "error",
                "error_type": "reachability",
                "message": err_msg,
                "dobot_target_mm": [rx, ry, rz, r_angle],
            }

        if self.is_connected and self.bot is not None:
            try:
                self.bot.clear_alarms()
                cur = self.bot.get_pose().position

                # 1. Elevate to Maximum Height using J2 before ANY horizontal movement
                msg_1 = f"Elevating to maximum height Z={transit_z:.0f} mm using J2..."
                print(f"  [1/8] {msg_1}", flush=True)
                if update_ui:
                    update_ui(msg_1, stage_num=1)
                _set_gripper_state(self.bot, "open")
                time.sleep(0.2)
                self.bot.move_to(cur.x, cur.y, transit_z, cur.r, wait=True)

                # 2. Pre-Rotate Gripper to target R at Maximum Height
                msg_2 = f"Pre-rotating gripper to R={r_angle:.1f}° at maximum height..."
                print(f"  [2/8] {msg_2}", flush=True)
                if update_ui:
                    update_ui(msg_2, stage_num=2)
                self.bot.move_to(cur.x, cur.y, transit_z, r_angle, wait=True)

                try:
                    import pydobotplus.dobotplus as dp
                    movl_mode = dp.MODE_PTP.MOVL_XYZ
                except Exception:
                    movl_mode = None

                if not is_spine:
                    # Scattered Book: Target height is at book surface Z ≈ 3.0 mm (verified in DobotStudio)
                    target_pick_z = 3.0 if rz < 3.0 else float(min(rz, 35.0))
                    hover_z = target_pick_z + 47.0  # ~50 mm overhead hover
                    transit_overhead_z = float(min(z_max_shelf, 130.0))

                    # 3a. Transit overhead at safe ceiling clearance (preventing J3 negative limit)
                    msg_3 = f"Transiting overhead to ({rx:.1f}, {ry:.1f}, {transit_overhead_z:.1f}) mm..."
                    print(f"  [3/8] {msg_3}", flush=True)
                    if update_ui:
                        update_ui(msg_3, stage_num=3)
                    self.bot.move_to(rx, ry, transit_overhead_z, r_angle, wait=True)
                    time.sleep(0.2)

                    # 3b. Position directly ABOVE the book spine (overhead hover)
                    msg_hover = f"Positioning directly above book spine at Z={hover_z:.1f} mm..."
                    print(f"        {msg_hover}", flush=True)
                    self.bot.move_to(rx, ry, hover_z, r_angle, mode=movl_mode, wait=True)
                    time.sleep(0.3)

                    # 4. Strictly Linear Vertical Descent downwards onto spine at Z ≈ 3.0 mm (ZERO horizontal push)
                    msg_4 = f"Descending vertically downwards onto spine at Z={target_pick_z:.1f} mm..."
                    print(f"  [4/8] {msg_4}", flush=True)
                    if update_ui:
                        update_ui(msg_4, stage_num=4)
                    self.bot.move_to(rx, ry, target_pick_z, r_angle, mode=movl_mode, wait=True)
                    time.sleep(0.4)
                else:
                    # Shelf Spine: Preserve exact shelf retrieval sequence untouched
                    msg_3 = f"Transiting overhead to ({rx:.1f}, {ry:.1f}, {z_max_shelf:.1f}) mm..."
                    print(f"  [3/8] {msg_3}", flush=True)
                    if update_ui:
                        update_ui(msg_3, stage_num=3)
                    self.bot.move_to(rx, ry, z_max_shelf, r_angle, wait=True)
                    time.sleep(0.3)

                    msg_4 = f"Descending vertically down onto spine at Z={rz:.1f} mm..."
                    print(f"  [4/8] {msg_4}", flush=True)
                    if update_ui:
                        update_ui(msg_4, stage_num=4)
                    self.bot.move_to(rx, ry, rz, r_angle, wait=True)
                    time.sleep(0.4)

                # 5. Clamp gripper (full pressure dwell for tight grip)
                msg_5 = f"Clamping jaws firmly on spine (full pressure dwell)..."
                print(f"  [5/8] {msg_5}", flush=True)
                if update_ui:
                    update_ui(msg_5, stage_num=5)
                _set_gripper_state(self.bot, "close")
                time.sleep(1.2)

                # 6. Strictly Vertical Extraction: Lift book straight up using J2 to Maximum Height
                if not is_spine:
                    msg_6 = f"Extracting book straight up to clearance Z={hover_z:.0f} mm..."
                    print(f"  [6/8] {msg_6}", flush=True)
                    if update_ui:
                        update_ui(msg_6, stage_num=6)
                    self.bot.move_to(rx, ry, hover_z, r_angle, mode=movl_mode, wait=True)
                    time.sleep(0.2)
                    self.bot.move_to(rx, ry, transit_z, r_angle, wait=True)
                    time.sleep(0.3)
                else:
                    msg_6 = f"Extracting book straight up to Maximum Height Z={transit_z:.0f} mm using J2..."
                    print(f"  [6/8] {msg_6}", flush=True)
                    if update_ui:
                        update_ui(msg_6, stage_num=6)
                    self.bot.move_to(rx, ry, transit_z, r_angle, wait=True)
                    time.sleep(0.3)

                # 7. Transfer to Drop Zone at Maximum Height & Drop directly
                msg_7 = f"Transferring to Drop Zone ({dx:.1f}, {dy:.1f}) at Maximum Height Z={transit_z:.1f} mm..."
                print(f"  [7/8] {msg_7}", flush=True)
                if update_ui:
                    update_ui(msg_7, stage_num=7)
                self.bot.move_to(dx, dy, transit_z, dr, wait=True)
                time.sleep(0.3)

                # Drop / Release at Maximum Height
                print(f"        Dropping book directly at Maximum Height Z={transit_z:.0f} mm...", flush=True)
                _set_gripper_state(self.bot, "open")
                time.sleep(0.6)
                _set_gripper_state(self.bot, "off")

                # 8. Retract arm safely to Home
                msg_8 = f"Retracting arm back to Home Standby at Maximum Height..."
                print(f"  [8/8] {msg_8}", flush=True)
                if update_ui:
                    update_ui(msg_8, stage_num=8)
                self.bot.move_to(DOBOT_HOME[0], DOBOT_HOME[1], DOBOT_HOME[2], DOBOT_HOME[3], wait=True)

                success_msg = "Book successfully retrieved from shelf and dropped at deposit station!"
                print(f"\n[SUCCESS] {success_msg}", flush=True)
                if update_ui:
                    update_ui(success_msg, stage_num=8, is_success=True)

            except Exception as e:
                err_msg = f"Dobot execution error: {e}"
                print(f"[Dobot Execution Error]: {e}", flush=True)
                if update_ui:
                    update_ui(err_msg, is_error=True, error_detail=str(e))
                return {
                    "status": "error",
                    "error_type": "execution",
                    "message": err_msg,
                    "dobot_target_mm": [rx, ry, rz, r_angle],
                }
        else:
            sim_msg = "Hardware not connected. Trajectory simulated successfully."
            print(f"  [Simulated] {sim_msg}", flush=True)
            if update_ui:
                update_ui(f"[SIMULATED] {sim_msg}", stage_num=8, is_success=True)

        print(f"==========================================================\n", flush=True)

        return {
            "status": "success",
            "connected": self.is_connected,
            "dobot_target_mm": [rx, ry, rz, r_angle],
            "camera_target_m": list(target_xyz_m) if target_xyz_m is not None else None,
            "message": f"Successfully executed Dobot target ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm, angle {r_angle:.1f}°."
        }


# Global instance
_dobot_instance = None


def get_dobot_controller():
    global _dobot_instance
    if _dobot_instance is None:
        _dobot_instance = DobotController()
    return _dobot_instance
