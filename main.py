"""
main.py
----------------------------------------------------------------------------
Complete Automated Book Retrieval Pipeline:

  query (voice/text)
    ──► capture RGB + depth (RealSense)
    ──► [1] mode classifier (tiny, always loaded, ~2ms)
    ──► [2] load/use ONLY the needed OBB model (Spine vs Scattered)
    ──► [3] OCR on OBB crops ──► match query ──► pick target
    ──► [4] SAM refine on the target only (sub-pixel exact edges)
    ──► [5] masked depth ──► deproject ──► XYZ (camera optical frame)
    ──► Step 6 transform ──► arm base frame (Dobot Magician)
"""

import os
import sys
import subprocess
from pathlib import Path

# Transparently re-launch inside GPU-enabled virtual environment if run via CPU interpreter (e.g. `py main.py`)
_VENV_PY = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
if __name__ == "__main__" and _VENV_PY.exists():
    try:
        _curr_exe = Path(sys.executable).resolve()
        _target_exe = _VENV_PY.resolve()
        if _curr_exe != _target_exe:
            import torch
            _has_cuda = torch.cuda.is_available()
        else:
            _has_cuda = True
    except Exception:
        _has_cuda = False

    if not _has_cuda:
        print(f"[Auto-GPU] Switching to CUDA virtual environment ({_VENV_PY})...", flush=True)
        res = subprocess.run([str(_VENV_PY)] + sys.argv, env=os.environ.copy())
        sys.exit(res.returncode)

import argparse

from startup import parallel_startup
from shared.mode_classifier import classify_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Book Retrieval Robot")
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "spine", "scattered", "mixed", "1", "2", "3", "4"],
        default="auto",
        help="Arrangement mode: auto (default), spine, scattered, mixed",
    )
    parser.add_argument(
        "--select", "-s",
        action="store_true",
        help="Interactively prompt for arrangement mode at startup",
    )
    parser.add_argument(
        "--pick", "-p",
        action="store_true",
        default=True,
        help="Execute Dobot pick sequence on detected book target (default: True)",
    )
    parser.add_argument(
        "--no-pick",
        action="store_true",
        help="Detect and localize target without executing physical robot pick",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable real-time OpenCV desktop monitor window",
    )
    # Parse only known args to safely allow extra runner flags
    args, _ = parser.parse_known_args()
    mode_map = {"1": "auto", "2": "spine", "3": "scattered", "4": "mixed"}
    args.mode = mode_map.get(args.mode, args.mode)
    return args


def run():
    args = parse_args()
    selected_mode = args.mode
    show_ui = not args.no_ui

    from shared.visual_monitor import VisualMonitor
    monitor = VisualMonitor(enabled=show_ui)
    if show_ui:
        monitor.show_startup("Listening for book query (speak anytime or type title)...")

    if args.select:
        print("\nSelect arrangement mode:")
        print("  [1] auto      - Automatically detect arrangement from camera (default)")
        print("  [2] spine     - Spines aligned on shelf")
        print("  [3] scattered - Books flat/scattered on surface")
        print("  [4] mixed     - Run both pipelines, pick best match")
        try:
            choice = input("Enter choice (1-4, default 1): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "1"
        choice_map = {
            "1": "auto", "auto": "auto", "a": "auto", "": "auto",
            "2": "spine", "spine": "spine", "s": "spine",
            "3": "scattered", "scattered": "scattered", "sc": "scattered",
            "4": "mixed", "mixed": "mixed", "m": "mixed",
        }
        selected_mode = choice_map.get(choice, "auto")
        print(f"Arrangement mode set to: '{selected_mode}'\n", flush=True)

    # Capture user query while camera and shared services warm up concurrently
    query_result, camera = parallel_startup()

    if query_result is None:
        print("\nNo book query matched. Exiting.")
        if camera:
            camera.stop()
        if show_ui:
            monitor.close()
        return None

    query_title = query_result["title"]
    print(f"\nTarget Query: '{query_title}'")
    if show_ui:
        monitor.update_status(f"Target query: '{query_title}'. Capturing RealSense RGB+Depth frame...", stage_num=0)

    cv_image = None
    target = None
    mode = selected_mode

    try:
        # Capture aligned color and depth frame from RealSense
        print("[Camera] Capturing aligned RGB + Depth frame...", flush=True)
        cv_image, depth_frame, color_intrinsics = camera.capture()
        if cv_image is None:
            print("\nFailed to capture frame from RealSense.")
            monitor.setup_target(None, None, query_title, mode="error")
            monitor.show_error("Camera Capture Failed", "RealSense D435i could not capture aligned color & depth frame.")
            monitor.wait_for_user(6000)
            return None

        # Display live camera frame immediately in the HUD monitor
        if show_ui:
            monitor.show_frame(cv_image, query_title=query_title, status="Camera frame acquired. Classifying arrangement mode...", mode=selected_mode)

        # Determine arrangement mode
        if selected_mode == "auto":
            # [1] Automatic Mode Classifier (tiny YOLO11n-cls + geometric spine verification)
            mode, mode_conf = classify_mode(cv_image, verbose=True)
            if show_ui:
                monitor.show_frame(cv_image, query_title=query_title, status=f"Scene classified as {mode.upper()} (conf={mode_conf:.2f}). Running {mode.upper()} pipeline...", mode=mode)
        else:
            mode = selected_mode
            print(f"[Mode] Using user-specified arrangement mode: '{mode.upper()}'", flush=True)
            if show_ui:
                monitor.show_frame(cv_image, query_title=query_title, status=f"Mode set to: '{mode.upper()}'. Running detection pipeline...", mode=mode)

        # [2] - [5] Load and execute the required pipeline on the pre-captured frame
        if mode == "spine":
            from spine_pipeline.find_target_spine import find_target_spine_live
            target = find_target_spine_live(
                query_title,
                camera=camera,
                cv_image=cv_image,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                refine_sam=True,
            )
        elif mode == "scattered":
            from scattered_pipeline.find_target_scattered import find_target_scattered_live
            target = find_target_scattered_live(
                query_title,
                camera=camera,
                cv_image=cv_image,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                refine_sam=True,
            )
        elif mode == "mixed":
            from mixed_pipeline.find_target_mixed import find_target_mixed_live
            target = find_target_mixed_live(
                query_title,
                camera=camera,
                cv_image=cv_image,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                refine_sam=True,
            )
        else:
            print(f"\nUnknown arrangement mode: '{mode}'")
            target = None

    finally:
        camera.stop()

    if target is None:
        print("\n=== NO TARGET FOUND ===")
        monitor.setup_target(cv_image, None, query_title, mode=mode)
        monitor.show_error(
            f"No target found matching query: '{query_title}'",
            "Candidates scored below similarity threshold (0.35). Check camera alignment and book lighting."
        )
        monitor.wait_for_user(7000)
        return None

    from shared.display_utils import print_target_summary
    print_target_summary(target)

    # Calculate robot coordinates and initialize visual UI monitor
    from robot.dobot_controller import get_dobot_controller, compute_parallel_rotation
    dobot = get_dobot_controller()
    pick_xyz_m = target.get("xyz_pick_camera_frame")
    if pick_xyz_m is None:
        pick_xyz_m = target.get("xyz_center_camera_frame")
    pick_angle = target.get("pick_edge_angle_deg")
    is_spine = (target.get("arrangement_mode") == "spine" or target.get("pipeline_winner") == "SPINE" or mode == "spine")

    robot_xyz = dobot.camera_to_dobot_coordinates(pick_xyz_m)
    rx, ry, rz = float(robot_xyz[0]), float(robot_xyz[1]), float(robot_xyz[2])
    r_angle = compute_parallel_rotation(rx, ry) if (is_spine or rz >= 60.0 or pick_angle is None) else float(pick_angle)
    robot_pose = [rx, ry, rz, r_angle]

    monitor.setup_target(cv_image, target, query_title, mode=mode, robot_coords=robot_pose)
    monitor.update_status(f"Target localized: '{target.get('text', query_title)}'. Ready for pick.", stage_num=0)

    # Step 6: Coordinate Transform from Optical Frame to Arm Base Frame & Physical Execution
    should_pick = args.pick and not args.no_pick
    if should_pick and target:
        def update_ui_cb(msg, stage_num=None, is_error=False, error_detail="", is_success=False):
            monitor.update_status(msg, stage_num=stage_num, is_error=is_error, error_detail=error_detail, is_success=is_success)

        pick_result = dobot.execute_pick_sequence(
            pick_xyz_m,
            grasp_angle_deg=pick_angle,
            is_spine=is_spine,
            update_ui=update_ui_cb
        )

        if pick_result.get("status") == "error":
            print("\n[Execution Stopped] Inspect monitor window for error diagnosis.", flush=True)
            monitor.wait_for_user(8000)
        else:
            monitor.wait_for_user(4000)

    return target


if __name__ == "__main__":
    run()
