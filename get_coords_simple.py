"""
get_coords_simple.py
-----------------------------------------------------------------------------
Normal, stable script to open RealSense camera and get 3D coordinates.

Features:
- Built-in RealSense Temporal + Spatial filters (ELIMINATES depth noise & jitter).
- Locks coordinates on click: values DO NOT flicker or change rapidly.
- Press SPACE to freeze/unfreeze frame to click precisely.
- Prints ONE clean, stable line per click.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# Load camera-to-robot calibration if available
try:
    import cam2robot as c2r
    has_calib = c2r.is_calibrated()
except Exception:
    c2r = None
    has_calib = False

# 1. Initialize RealSense Camera
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Depth Filters to stop rapid jitter/fluctuation
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

print("\n" + "=" * 60)
print(" CAMERA OPENED - NORMAL STABLE COORDINATE VIEWER")
print("=" * 60)
print(" Left Click : Click any point to lock coordinates (STABLE, NO JITTER)")
print(" SPACE      : Freeze / Unfreeze image")
print(" Q / ESC    : Quit")
print("=" * 60 + "\n")

# State
locked_point = None
locked_info = None
is_frozen = False
frozen_color = None
frozen_depth = None


def get_stable_depth(depth_frame_data, u, v, radius=4):
    """Samples depth in a patch and returns stable median."""
    u1, u2 = max(0, u - radius), min(640, u + radius + 1)
    v1, v2 = max(0, v - radius), min(480, v + radius + 1)
    patch = depth_frame_data[v1:v2, u1:u2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def mouse_callback(event, x, y, flags, param):
    global locked_point, locked_info
    if event == cv2.EVENT_LBUTTONDOWN:
        u, v = x, y
        active_depth = frozen_depth if is_frozen else current_depth

        d_m = get_stable_depth(active_depth, u, v)
        if d_m <= 0:
            print(f"[Warning] No depth returned at pixel ({u}, {v}). Try clicking slightly inside the book edge.")
            return

        # Deproject to Camera 3D (mm)
        cam_xyz_m = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], d_m)
        cx, cy, cz = [round(c * 1000.0, 1) for c in cam_xyz_m]
        d_mm = round(d_m * 1000.0, 1)

        # Robot Base 3D
        rx, ry, rz = None, None, None
        j1_deg = None
        if has_calib and c2r:
            try:
                rob = c2r.cam_to_robot(cam_xyz_m)
                rx, ry, rz = round(float(rob[0]), 1), round(float(rob[1]), 1), round(float(rob[2]), 1)
                j1_deg = round(float(np.degrees(np.arctan2(ry, rx))), 1)
            except Exception:
                pass

        # LOCK IT SO IT DOES NOT CHANGE
        locked_point = (u, v)
        locked_info = {
            "u": u, "v": v,
            "depth_mm": d_mm,
            "cam": (cx, cy, cz),
            "rob": (rx, ry, rz) if rx is not None else None,
            "j1": j1_deg,
        }

        # Print ONE clean line
        if locked_info["rob"]:
            print(f">>> Pixel: ({u}, {v}) | Depth: {d_mm} mm | Cam 3D: {cx}, {cy}, {cz} mm | Robot: X={rx}, Y={ry}, Z={rz} mm | J1: {j1_deg} deg")
        else:
            print(f">>> Pixel: ({u}, {v}) | Depth: {d_mm} mm | Cam 3D: X={cx}, Y={cy}, Z={cz} mm")


cv2.namedWindow("Camera Coordinates", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Camera Coordinates", mouse_callback)

try:
    while True:
        if not is_frozen:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()

            if not c_frame or not d_frame:
                continue

            # Apply filters for rock-solid depth
            d_filtered = spatial.process(d_frame)
            d_filtered = temporal.process(d_filtered)
            d_filtered = hole_filling.process(d_filtered)

            current_color = np.asanyarray(c_frame.get_data())
            current_depth = np.asanyarray(d_filtered.get_data()) * depth_scale
            display_img = current_color.copy()
        else:
            display_img = frozen_color.copy()

        # Draw locked target if clicked
        if locked_point and locked_info:
            u, v = locked_point
            cv2.circle(display_img, (u, v), 5, (0, 0, 255), -1)
            cv2.circle(display_img, (u, v), 10, (0, 255, 255), 2)
            cv2.line(display_img, (u - 15, v), (u + 15, v), (0, 255, 255), 1)
            cv2.line(display_img, (u, v - 15), (u, v + 15), (0, 255, 255), 1)

            # Static, non-flickering text box
            box_text = f"P:({u},{v}) | D:{locked_info['depth_mm']}mm"
            cv2.putText(display_img, box_text, (max(10, u - 60), max(25, v - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if locked_info["rob"]:
                rx, ry, rz = locked_info["rob"]
                rob_text = f"Rob: [{rx}, {ry}, {rz}] mm | J1: {locked_info['j1']}deg"
                cv2.putText(display_img, rob_text, (max(10, u - 60), max(45, v + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Status text
        status = "[PAUSED / FROZEN - click freely]" if is_frozen else "[LIVE FEED - SPACE to freeze]"
        cv2.putText(display_img, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Camera Coordinates", display_img)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # SPACE
            is_frozen = not is_frozen
            if is_frozen:
                frozen_color = current_color.copy()
                frozen_depth = current_depth.copy()
                print("[Frame FROZEN] You can now click cleanly on the book without any motion.")
            else:
                print("[Frame UNPAUSED - Live]")

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("[Camera closed]")
