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
    # Parse only known args to safely allow extra runner flags
    args, _ = parser.parse_known_args()
    mode_map = {"1": "auto", "2": "spine", "3": "scattered", "4": "mixed"}
    args.mode = mode_map.get(args.mode, args.mode)
    return args


def run():
    args = parse_args()
    selected_mode = args.mode

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
        return None

    query_title = query_result["title"]
    print(f"\nTarget Query: '{query_title}'")

    try:
        # Capture aligned color and depth frame from RealSense
        print("[Camera] Capturing aligned RGB + Depth frame...", flush=True)
        cv_image, depth_frame, color_intrinsics = camera.capture()
        if cv_image is None:
            print("\nFailed to capture frame from RealSense.")
            return None

        # Determine arrangement mode
        if selected_mode == "auto":
            # [1] Automatic Mode Classifier (tiny YOLO11n-cls + geometric spine verification)
            mode, mode_conf = classify_mode(cv_image, verbose=True)
        else:
            mode = selected_mode
            print(f"[Mode] Using user-specified arrangement mode: '{mode.upper()}'", flush=True)

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
        return None

    from shared.display_utils import print_target_summary
    print_target_summary(target)

    # Step 6: Coordinate Transform from Optical Frame to Arm Base Frame & Physical Execution
    should_pick = args.pick and not args.no_pick
    if should_pick and target:
        pick_xyz_m = target.get("xyz_pick_camera_frame") or target.get("xyz_center_camera_frame")
        pick_angle = target.get("pick_edge_angle_deg")
        is_spine = (mode == "spine")
        from robot.dobot_controller import get_dobot_controller
        dobot = get_dobot_controller()
        dobot.execute_pick_sequence(pick_xyz_m, grasp_angle_deg=pick_angle, is_spine=is_spine)

    return target


if __name__ == "__main__":
    run()
