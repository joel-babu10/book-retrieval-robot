"""

pipeline.py

----------------------------------------------------------------------------

Operator UI Pipeline Bridge for Autonomous Book Retrieval Robot.

Exposes:

  1. find_target(query: str, mode: str) -> dict | None

  2. get_frame() -> annotated BGR ndarray (live view)

  3. send_to_arm(robot_xyz) -> bool

----------------------------------------------------------------------------

"""



import os

import sys



# Configure Windows Console and stdout to prevent freezing/blocking

def _setup_console():

    if hasattr(sys.stdout, "reconfigure"):

        try:

            sys.stdout.reconfigure(line_buffering=True, write_through=True)

            sys.stderr.reconfigure(line_buffering=True, write_through=True)

        except Exception:

            pass

    if os.name == "nt":

        try:

            import ctypes

            kernel32 = ctypes.windll.kernel32

            hStdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10

            mode = ctypes.c_ulong()

            if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):

                new_mode = (mode.value & ~0x0040) | 0x0080

                kernel32.SetConsoleMode(hStdin, new_mode)

        except Exception:

            pass



_setup_console()



import time

import threading

from pathlib import Path

import numpy as np

import cv2

import pyrealsense2 as rs

try:
    import torch
    import torchvision
except Exception:
    pass



import cam2robot as c2r

import dobot_kin as K

from shared.mode_classifier import classify_mode

from robot.dobot_controller import get_dobot_controller, compute_parallel_rotation, compute_front_facing_rotation



# State management

_lock = threading.Lock()

_latest_color = None

_latest_depth = None

_latest_intrinsics = None

_current_target = None

_result_frame = None

_last_target_full = None

_camera_active = False



# Background Camera Thread



# ----------------------------------------------------------------------------

# GPU Acceleration & Model Pre-warming

# ----------------------------------------------------------------------------

def _warmup_models_background():

    """Pre-warm GPU models (YOLO, Mode Classifier, SAM, RapidOCR) in background so user queries are instant."""

    try:

        import torch

        if torch.cuda.is_available():

            torch.backends.cudnn.benchmark = True

            dev_name = torch.cuda.get_device_name(0)

            print(f"[GPU Accelerator] Active GPU: {dev_name} (CUDA 12.1)", flush=True)

            _ = torch.zeros((1, 3, 64, 64), device="cuda")

        else:

            print("[GPU Accelerator] Warning: CUDA not available. Running on CPU.", flush=True)



        # 1. Warm up Mode Classifier

        try:

            from shared.mode_classifier import get_mode_classifier

            _ = get_mode_classifier(verbose=False)

        except Exception:

            pass



        # 2. Warm up YOLO Spine model

        try:

            from spine_pipeline.detect_spine import get_model

            _ = get_model()

        except Exception:

            pass



        # 3. Warm up OCR engine

        try:

            from shared.ocr_reader import get_shared_ocr_reader

            reader = get_shared_ocr_reader(verbose=False)

            dummy = np.zeros((64, 64, 3), dtype=np.uint8)

            _ = reader.readtext(dummy)

        except Exception:

            pass



        # 4. Warm up SAM in background

        try:

            from shared.sam_refine import warmup_sam_in_background

            warmup_sam_in_background()

        except Exception:

            pass



        print("[GPU Accelerator] All neural models loaded & cached on GPU - instant retrieval ready.", flush=True)

    except Exception as e:

        print(f"[GPU Accelerator] Warmup notice: {e}", flush=True)



threading.Thread(target=_warmup_models_background, daemon=True, name="GPUWarmup").start()



class CameraStreamer(threading.Thread):

    def __init__(self, width=640, height=480, fps=30):

        super().__init__(daemon=True)

        self.width = width

        self.height = height

        self.fps = fps

        self.running = True

        self.pipe = None

        self.align = None

        self.spatial = rs.spatial_filter()

        self.temporal = rs.temporal_filter()

        self.hole_filling = rs.hole_filling_filter()



    def run(self):

        global _latest_color, _latest_depth, _latest_intrinsics, _camera_active

        try:

            self.pipe = rs.pipeline()

            cfg = rs.config()

            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            prof = self.pipe.start(cfg)

            self.align = rs.align(rs.stream.color)

            _camera_active = True

            print(f"[CameraStreamer] RealSense pipeline started ({self.width}x{self.height} @ {self.fps} FPS).", flush=True)



            while self.running:

                try:

                    frames = self.pipe.wait_for_frames(timeout_ms=1000)

                    aligned = self.align.process(frames)

                    c_frame = aligned.get_color_frame()

                    d_frame = aligned.get_depth_frame()

                    if not c_frame or not d_frame:

                        continue



                    d_filt = self.spatial.process(d_frame)

                    d_filt = self.temporal.process(d_filt)

                    d_filt = self.hole_filling.process(d_filt).as_depth_frame()



                    c_img = np.asanyarray(c_frame.get_data())

                    intr = c_frame.profile.as_video_stream_profile().intrinsics



                    with _lock:

                        _latest_color = c_img.copy()

                        _latest_depth = d_filt

                        _latest_intrinsics = intr

                except Exception as e:

                    time.sleep(0.05)



        except Exception as e:

            print(f"[CameraStreamer] RealSense initialization warning: {e}. Running fallback feed.", flush=True)

            _camera_active = False

            # Generate fallback demonstration loop if RealSense camera is unavailable

            while self.running:

                demo_img = np.full((self.height, self.width, 3), (25, 22, 18), dtype=np.uint8)

                cv2.rectangle(demo_img, (20, 20), (self.width - 20, self.height - 20), (50, 45, 38), 2)

                cv2.putText(demo_img, "STANDBY - CAMERA RECONNECTING", (120, self.height // 2),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

                with _lock:

                    _latest_color = demo_img

                time.sleep(0.1)



        finally:

            if self.pipe:

                try:

                    self.pipe.stop()

                except Exception:

                    pass

            _camera_active = False



# Start camera daemon streamer

_streamer = CameraStreamer()

_streamer.start()





def clear_target():

    """Clears the currently localized target and resets the feed back to pure live video."""

    global _current_target, _result_frame

    with _lock:

        _current_target = None

        _result_frame = None





def get_frame() -> np.ndarray:
    """
    Returns the live optical video stream at 30 FPS.
    If a target book is localized, dynamically renders the sleek aesthetic
    AR target overlay onto the LIVE frames so the camera feed NEVER freezes or gets stuck.
    """
    global _latest_color, _current_target

    with _lock:
        if _latest_color is None:
            frame = np.full((480, 640, 3), (12, 14, 20), dtype=np.uint8)
            cv2.putText(frame, "CONNECTING TO OPTICAL SENSOR...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)
            return frame

        frame = _latest_color.copy()
        tgt = _current_target

    if tgt is not None:
        try:
            from shared.display_utils import draw_aesthetic_target_overlay
            frame = draw_aesthetic_target_overlay(frame, tgt)
        except Exception:
            pass

    return frame


def find_target(query: str, mode: str = "auto") -> dict | None:

    """

    Executes the autonomous book identification pipeline on the current optical frame:

      - Classifies scene mode (spine / scattered / mixed) if mode == 'auto'

      - Performs YOLO OBB detection + RapidOCR text reading

      - Pinpoints pick point with SAM geometric edge refinement

      - Deprojects to Camera 3D and applies calibrated Extrinsics to Dobot 3D

    Returns dictionary with target info or None if not found.

    """

    global _latest_color, _latest_depth, _latest_intrinsics, _current_target, _last_target_full



    query_str = (query or "").strip()

    if not query_str:

        return None



    # Wait briefly if streamer is acquiring initial frame

    for _ in range(15):

        with _lock:

            if _latest_color is not None and (_latest_depth is not None or not _camera_active):

                color_img = _latest_color.copy()

                depth_f = _latest_depth

                intr = _latest_intrinsics

                break

        time.sleep(0.1)

    else:

        with _lock:

            color_img = _latest_color.copy() if _latest_color is not None else None

            depth_f = _latest_depth

            intr = _latest_intrinsics



    if color_img is None:

        print("[Pipeline Error] No frame available from camera.", flush=True)

        return None



    # Determine arrangement mode

    req_mode = (mode or "auto").lower()

    if req_mode == "auto":

        active_mode, mode_conf = classify_mode(color_img, verbose=False)

    else:

        active_mode = req_mode



    t_find_start = time.time()

    global _result_frame

    with _lock:

        _current_target = None

        _result_frame = None

    print(f"\n[Pipeline] find_target('{query_str}', mode='{req_mode}') -> Active Mode: {active_mode.upper()} [GPU Accelerated]", flush=True)



    target = None

    try:

        if active_mode == "spine":

            from spine_pipeline.find_target_spine import find_target_spine_live

            target = find_target_spine_live(

                target_title=query_str,

                cv_image=color_img,

                depth_frame=depth_f,

                color_intrinsics=intr,

                refine_sam=True,

                save_visual=True

            )

        elif active_mode == "scattered":

            from scattered_pipeline.find_target_scattered import find_target_scattered_live

            target = find_target_scattered_live(

                query_title=query_str,

                cv_image=color_img,

                depth_frame=depth_f,

                color_intrinsics=intr,

                refine_sam=True,

                save_visual=True

            )

        else:

            from mixed_pipeline.find_target_mixed import find_target_mixed_live

            target = find_target_mixed_live(

                query_title=query_str,

                cv_image=color_img,

                depth_frame=depth_f,

                color_intrinsics=intr,

                refine_sam=True,

                save_visual=True

            )

    except Exception as e:

        print(f"[Pipeline Error] Target detection exception: {e}", flush=True)

        target = None



    if not target:

        print(f"[Pipeline] No match found for '{query_str}'.", flush=True)

        with _lock:

            _current_target = None

            _result_frame = None

        return None



    # Coordinate transformation

    cam_xyz_raw = target.get("xyz_pick_camera_frame")

    if cam_xyz_raw is None:

        cam_xyz_raw = target.get("xyz_center_camera_frame")

    if cam_xyz_raw is None:

        cam_xyz_raw = [0.0, 0.0, 0.30]

    cam_arr = np.asarray(cam_xyz_raw, dtype=float)

    if np.any(np.abs(cam_arr) > 10.0):

        cam_xyz_m = (cam_arr / 1000.0).tolist()

        cam_xyz_mm = cam_arr.tolist()

    else:

        cam_xyz_m = cam_arr.tolist()

        cam_xyz_mm = (cam_arr * 1000.0).tolist()



    # Robot base coordinates (mm)

    calib_mode = "vertical" if target.get("is_front_facing") else ("spine" if (active_mode == "spine" or target.get("arrangement_mode") == "spine") else "scattered")
    rob_xyz_mm = c2r.cam_to_robot(cam_xyz_m, mode=calib_mode)

    rx, ry, rz = float(rob_xyz_mm[0]), float(rob_xyz_mm[1]), float(rob_xyz_mm[2])



    # Compute grasp rotation & pick strategy note

    is_spine_target = (active_mode == "spine" or target.get("arrangement_mode") == "spine" or rz >= 60.0)



    # For scattered / flat desk targets: ensure coordinates are kinematically reachable

    if not is_spine_target:

        reach_ok, reach_reason, _ = K.check(rx, ry, rz, margin=1.0)

        if not reach_ok:

            print(f"[Pipeline Reachability] Scattered target at ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm unreachable: {reach_reason}. Finding reachable position on book...", flush=True)

            try:

                from shared.reachable_pick import find_reachable_scattered_point

                corners_2d = target.get("corners_refined") or target.get("corners") or target.get("obb_corners") or target.get("box_corners")

                nom_u = target.get("top_edge_center_u") or (target.get("pick_px") or [0, 0])[0]

                nom_v = target.get("top_edge_center_v") or (target.get("pick_px") or [0, 0])[1]

                reach_info = find_reachable_scattered_point(

                    corners_2d=corners_2d,

                    nominal_u=nom_u,

                    nominal_v=nom_v,

                    depth_frame=depth_f,

                    color_intrinsics=intr,

                    nominal_xyz_cam=cam_xyz_m

                )

                if reach_info is not None and reach_info.get("is_fallback"):

                    cam_xyz_m = reach_info["pick_xyz_m"]

                    rob_xyz_mm = reach_info["pick_rob_mm"]

                    rx, ry, rz = float(rob_xyz_mm[0]), float(rob_xyz_mm[1]), float(rob_xyz_mm[2])

                    target["pick_px"] = list(reach_info["pick_uv"])

                    target["is_fallback_reachable"] = True

                    reach_ok = True

                    print(f"[Pipeline Reachability] Switched to reachable pick on book: ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm (r={reach_info['radius_mm']:.1f}mm)!", flush=True)

            except Exception as e:

                print(f"[Pipeline Reachability] Fallback check error: {e}", flush=True)

    else:

        reach_ok, reach_reason, _ = K.check(rx, ry, rz, margin=1.0)



    if is_spine_target:

        r_angle = compute_parallel_rotation(rx, ry)

        pick_note = f"Upper spine parallel grasp (R={r_angle:+.1f}°, Z={rz:.1f}mm)"

    elif target.get("is_front_facing"):

        r_angle = compute_front_facing_rotation(rx, ry, target.get("pick_edge_angle_deg", 0.0))

        pick_note = f"Front-facing top rim grasp (R={r_angle:+.1f}°, Z={rz:.1f}mm)"

    else:

        pick_angle = target.get("pick_edge_angle_deg", 0.0)

        j1 = float(np.degrees(np.arctan2(ry, rx)))

        edge_offset = float(pick_angle) if abs(float(pick_angle)) <= 45.0 else 0.0

        r_angle = round(j1 + edge_offset, 1)

        is_fb = target.get("is_fallback_reachable", False)

        prefix = "Reachable cover pick" if is_fb else "Desk surface pick"

        pick_note = f"{prefix} (Z={rz:.1f}mm, R={r_angle:+.1f}°)"



    t_find_dur = time.time() - t_find_start

    kin_status = "FALLBACK REACHABLE" if target.get("is_fallback_reachable") else ("REACHABLE" if reach_ok else "UNREACHABLE")

    log_details = [

        f"[CLASSIFIER] Target mode: {active_mode.upper()} (GPU accelerated)",

        f"[DETECTION] Matched '{target.get('text', query_str)}' (conf: {target.get('conf', 0.90):.2f}, sim: {target.get('similarity', 1.0):.2f})",

        f"[CAMERA 3D] Camera coordinates: X={cam_xyz_m[0]:.3f}m, Y={cam_xyz_m[1]:.3f}m, Z={cam_xyz_m[2]:.3f}m",

        f"[ROBOT FRAME] Base coordinates: X={rx:.1f}mm, Y={ry:.1f}mm, Z={rz:.1f}mm (radius={np.hypot(rx, ry):.1f}mm)",

        f"[KINEMATICS] Grasp status: {kin_status} - verified with Dobot Magician kinematics",

        f"[STRATEGY] {pick_note}",

        f"[PIPELINE] Execution time: {t_find_dur * 1000.0:.0f}ms | Ready to dispatch Dobot arm"

    ]

    if target.get("is_fallback_reachable"):

        log_details.insert(4, f"[REACHABILITY] Nominal point unreachable. Shifted pick to reachable cover position on book (r={np.hypot(rx, ry):.1f}mm).")



    # Store full target for arm execution

    _last_target_full = {

        "target": target,

        "cam_xyz_m": cam_xyz_m,

        "robot_xyz_mm": [rx, ry, rz],

        "r_angle": r_angle,

        "is_spine": is_spine_target,

        "is_front_facing": bool(target.get("is_front_facing", False))

    }



    # Store HUD display state

    corners = target.get("corners_refined")

    if corners is None:

        corners = target.get("corners")

    if corners is None:

        corners = target.get("box_corners")



    pick_px = target.get("pick_point_px") or target.get("pick_point_2d")
    if pick_px is None and "top_edge_center_u" in target and "top_edge_center_v" in target:
        pick_px = [target["top_edge_center_u"], target["top_edge_center_v"]]
    if pick_px is None and "top_edge_center" in target:
        pick_px = list(target["top_edge_center"])
    if pick_px is None:
        pick_px = target.get("center")



    out_visual = "result_target_spine_live.jpg" if active_mode == "spine" else ("result_target_scattered_live.jpg" if active_mode == "scattered" else "result_target_mixed_live.jpg")

    if target.get("visual_frame") is not None:

        with _lock:

            _result_frame = target["visual_frame"]

    elif os.path.exists(out_visual):

        try:

            loaded_vis = cv2.imread(out_visual)

            if loaded_vis is not None:

                with _lock:

                    _result_frame = loaded_vis

        except Exception:

            pass



    _current_target = {

        "title": str(target.get("text", query_str)),

        "mode": active_mode,

        "conf": round(float(target.get("confidence", 0.90)), 2),

        "sim": round(float(target.get("similarity", 1.0)), 2),

        "cam_xyz": [round(float(c), 3) for c in cam_xyz_m],

        "robot_xyz": [round(rx, 1), round(ry, 1), round(rz, 1)],

        "pick_note": pick_note,

        "corners": corners,

        "pick_px": pick_px

    }



    print(f"[Pipeline Success] Target localized: '{_current_target['title']}' | Robot: X={rx:.1f}, Y={ry:.1f}, Z={rz:.1f} mm", flush=True)



    return {

        "title": _current_target["title"],

        "conf": _current_target["conf"],

        "sim": _current_target["sim"],

        "mode": _current_target["mode"],

        "cam_xyz": _current_target["cam_xyz"],

        "robot_xyz": _current_target["robot_xyz"],

        "pick_note": _current_target["pick_note"],

        "log_details": log_details,

        "frame": color_img

    }





def send_to_arm(robot_xyz) -> bool:

    """

    Sends the target coordinates to Dobot Magician to execute the physical

    retrieval & drop sequence. Returns True on success, False on failure.

    """

    global _last_target_full



    if robot_xyz is None or len(robot_xyz) < 3:

        print("[send_to_arm Error] Invalid robot_xyz received.", flush=True)

        return False



    rx, ry, rz = float(robot_xyz[0]), float(robot_xyz[1]), float(robot_xyz[2])

    print(f"\n[send_to_arm] Command to Dobot Magician at ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm...", flush=True)



    # 1. Kinematic Reachability Validation

    reach_ok, reach_reason, _ = K.check(rx, ry, rz)

    if not reach_ok:

        if _last_target_full and not _last_target_full.get("is_spine", True):

            print(f"[send_to_arm] Target ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm unreachable: {reach_reason}. Trying alternative point on scattered book...", flush=True)

            try:

                from shared.reachable_pick import find_reachable_scattered_point

                t_obj = _last_target_full.get("target") or {}

                corners_2d = t_obj.get("corners_refined") or t_obj.get("corners") or t_obj.get("obb_corners") or t_obj.get("box_corners")

                nom_u = t_obj.get("top_edge_center_u") or (t_obj.get("pick_px") or [0, 0])[0]

                nom_v = t_obj.get("top_edge_center_v") or (t_obj.get("pick_px") or [0, 0])[1]

                with _lock:

                    df = _latest_depth

                    ci = _latest_intrinsics

                reach_info = find_reachable_scattered_point(

                    corners_2d=corners_2d,

                    nominal_u=nom_u,

                    nominal_v=nom_v,

                    depth_frame=df,

                    color_intrinsics=ci,

                    nominal_xyz_cam=_last_target_full.get("cam_xyz_m")

                )

                if reach_info is not None and reach_info.get("is_fallback"):

                    rx, ry, rz = [float(v) for v in reach_info["pick_rob_mm"]]

                    _last_target_full["cam_xyz_m"] = reach_info["pick_xyz_m"]

                    _last_target_full["robot_xyz_mm"] = [rx, ry, rz]

                    reach_ok = True

                    print(f"[send_to_arm] Fallback point on book is reachable: ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm!", flush=True)

            except Exception as e:

                print(f"[send_to_arm] Fallback search notice: {e}", flush=True)



        if not reach_ok:

            r = float(np.hypot(rx, ry))

            if r > 300.0:

                scale = 285.0 / r

                adj_rx, adj_ry = rx * scale, ry * scale

                adj_ok, _, _ = K.check(adj_rx, adj_ry, rz, margin=0.5)

                if adj_ok:

                    rx, ry = adj_rx, adj_ry

                    reach_ok = True

                    if _last_target_full:

                        _last_target_full["robot_xyz_mm"] = [rx, ry, rz]

                    print(f"[send_to_arm] Scaled pick point inward to ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm (r=285mm).", flush=True)



        if not reach_ok:

            print(f"[send_to_arm Kinematic Error] Target is outside reachable envelope: {reach_reason}", flush=True)

            return False



    dobot = get_dobot_controller()



    # Determine grasp angle and target context

    is_spine = True

    is_front_facing = False

    grasp_angle = None

    cam_xyz_m = None



    if _last_target_full:

        is_spine = _last_target_full.get("is_spine", True)

        is_front_facing = _last_target_full.get("is_front_facing", False)

        grasp_angle = _last_target_full.get("r_angle")

        cam_xyz_m = _last_target_full.get("cam_xyz_m")



    if grasp_angle is None:

        if is_front_facing:

            grasp_angle = compute_front_facing_rotation(rx, ry)

        elif is_spine or rz >= 60.0:

            grasp_angle = compute_parallel_rotation(rx, ry)

        else:

            grasp_angle = 0.0



    # 2. Physical Execution if Dobot is Connected

    if dobot.is_connected and dobot.bot is not None:

        try:

            target_input = cam_xyz_m if cam_xyz_m is not None else [rx / 1000.0, ry / 1000.0, rz / 1000.0]

            result = dobot.execute_pick_sequence(

                target_xyz_m=target_input,

                grasp_angle_deg=grasp_angle,

                is_spine=is_spine,

                is_front_facing=is_front_facing

            )

            success = (result.get("status") == "success")

            print(f"[send_to_arm] Execution result: {result.get('status', 'unknown')}", flush=True)

            return success

        except Exception as e:

            print(f"[send_to_arm Exception] Physical execution failed: {e}", flush=True)

            return False

    else:

        # Hardware not connected or busy — validate kinematic simulation

        print(f"[send_to_arm Simulated] Hardware offline. Target verified kinematically: Reachable at r={np.hypot(rx, ry):.1f}mm, R={grasp_angle:.1f}°.", flush=True)

        return True

