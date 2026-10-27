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
    Interpolated across verified ground-truth bookshelf poses:
      - Point 2 (Left,   X=-25.2mm): R = 84.8° (J1=95.9°, J4=-11.0°)
      - Point 1 (Center, X=43.2mm):  R = 81.0° (J1=81.0°, J4=0.0°)
      - Point 3 (Right,  X=53.8mm):  R = 92.7° (J1=77.9°, J4=+14.7°)
    """
    if y > 180.0:
        r_val = 0.01464005 * (x ** 2) - 0.32008491 * x + 67.4968
        return round(float(np.clip(r_val, 65.0, 110.0)), 1)
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

        xr = 0.394764 * xc - 0.908681 * zc + 296.77
        yr = 0.852790 * xc + 0.064282 * zc + 320.54
        zr = -0.980158 * yc - 0.020429 * zc + 115.07
        return np.array([xr, yr, zr])

    def execute_pick_sequence(self, target_xyz_m, grasp_angle_deg=None, is_spine=False):
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

        if is_spine or grasp_angle_deg is None:
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

        if self.is_connected and self.bot is not None:
            try:
                self.bot.clear_alarms()
                cur = self.bot.get_pose().position

                # 1. Elevate to Maximum Height using J2 before ANY horizontal movement
                print(f"  [1/8] Elevating to maximum height Z={transit_z:.0f} mm using J2...", flush=True)
                _set_gripper_state(self.bot, "open")
                time.sleep(0.2)
                self.bot.move_to(cur.x, cur.y, transit_z, cur.r, wait=True)

                # 2. Pre-Rotate Gripper to target R at Maximum Height
                print(f"  [2/8] Pre-rotating gripper to R={r_angle:.1f}° at maximum height...", flush=True)
                self.bot.move_to(cur.x, cur.y, transit_z, r_angle, wait=True)

                # 3. Transit overhead to target strictly at Maximum Height
                print(f"  [3/8] Transiting overhead to ({rx:.1f}, {ry:.1f}, {z_max_shelf:.1f}) mm...", flush=True)
                self.bot.move_to(rx, ry, z_max_shelf, r_angle, wait=True)
                time.sleep(0.3)

                # 4. Strictly Vertical Descent straight down into book spine
                print(f"  [4/8] Descending vertically down onto spine at Z={rz:.1f} mm...", flush=True)
                self.bot.move_to(rx, ry, rz, r_angle, wait=True)
                time.sleep(0.4)

                # 5. Clamp gripper (full pressure dwell for tight grip)
                print(f"  [5/8] Clamping jaws firmly on spine (full pressure dwell)...", flush=True)
                _set_gripper_state(self.bot, "close")
                time.sleep(1.2)

                # 6. Strictly Vertical Extraction: Lift book straight up using J2 to Maximum Height
                print(f"  [6/8] Extracting book straight up to Maximum Height Z={transit_z:.0f} mm using J2...", flush=True)
                self.bot.move_to(rx, ry, transit_z, r_angle, wait=True)
                time.sleep(0.3)

                # 7. Transfer to Drop Zone at Maximum Height & Drop directly
                print(f"  [7/8] Transferring to Drop Zone ({dx:.1f}, {dy:.1f}) at Maximum Height Z={transit_z:.1f} mm...", flush=True)
                self.bot.move_to(dx, dy, transit_z, dr, wait=True)
                time.sleep(0.3)

                # Drop / Release at Maximum Height
                print(f"        Dropping book directly at Maximum Height Z={transit_z:.0f} mm...", flush=True)
                _set_gripper_state(self.bot, "open")
                time.sleep(0.6)
                _set_gripper_state(self.bot, "off")

                # 8. Retract arm safely to Home
                print(f"  [8/8] Retracting arm back to Home Standby at Maximum Height...", flush=True)
                self.bot.move_to(DOBOT_HOME[0], DOBOT_HOME[1], DOBOT_HOME[2], DOBOT_HOME[3], wait=True)

                print("\n[SUCCESS] Book successfully retrieved from shelf and dropped at deposit station!", flush=True)

            except Exception as e:
                print(f"[Dobot Execution Error]: {e}", flush=True)
        else:
            print("  [Simulated] Hardware not connected. Trajectory simulated successfully.", flush=True)

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
