"""
main.py
----------------------------------------------------------------------------
Autonomous Book Retrieval Robot — Clean Terminal Interface
----------------------------------------------------------------------------
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# Re-launch inside GPU-enabled virtual environment if run via CPU interpreter
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
        res = subprocess.run([str(_VENV_PY)] + sys.argv, env=os.environ.copy())
        sys.exit(res.returncode)


def get_user_query(cli_query=None):
    """
    Acquires the target book query from CLI argument or clean interactive terminal prompt.
    Returns (result_dict, match_score, source).
    """
    from input.query_input import VOICE_AVAILABLE, match_query, load_catalog

    catalog = load_catalog()

    if cli_query and cli_query.strip():
        query_str = cli_query.strip()
        result, score = match_query(query_str, catalog=catalog)
        return result, score, "cli"

    print("\nRequest a Book:")
    if VOICE_AVAILABLE:
        print("  • Type title or author and press Enter")
        print("  • Or press Enter without typing to speak (voice)")
    else:
        print("  • Type title or author and press Enter")

    try:
        query_str = input("\nEnter book title: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None, 0.0, "cancel"

    source = "text"
    if not query_str:
        if VOICE_AVAILABLE:
            print("  [Voice Mode] Listening... Please speak clearly now:", flush=True)
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                recognizer.dynamic_energy_threshold = False
                recognizer.energy_threshold = 300
                with sr.Microphone() as mic:
                    recognizer.adjust_for_ambient_noise(mic, duration=0.8)
                    audio = recognizer.listen(mic, timeout=6, phrase_time_limit=10)
                    query_str = recognizer.recognize_google(audio, language="en-US")
                    source = "voice"
                    print(f"  [Voice Recognized] \"{query_str}\"", flush=True)
            except Exception:
                print("  [Notice] No speech detected or voice recognition unavailable.", flush=True)
                return None, 0.0, "voice_fail"
        else:
            return None, 0.0, "empty"

    result, score = match_query(query_str, catalog=catalog)
    return result, score, source


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Book Retrieval Robot")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Target book title or query (skips interactive prompt)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "spine", "scattered", "mixed", "1", "2", "3", "4"],
        default="auto",
        help="Arrangement mode: auto (default), spine, scattered, mixed",
    )
    parser.add_argument(
        "--select", "-s",
        action="store_true",
        help="Interactively select arrangement mode",
    )
    parser.add_argument(
        "--pick", "-p",
        action="store_true",
        default=True,
        help="Execute Dobot pick sequence (default: True)",
    )
    parser.add_argument(
        "--no-pick",
        action="store_true",
        help="Localize target without executing arm movement",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Display internal debug and candidate detection logs",
    )
    args, _ = parser.parse_known_args()
    mode_map = {"1": "auto", "2": "spine", "3": "scattered", "4": "mixed"}
    args.mode = mode_map.get(args.mode, args.mode)
    return args


def run():
    args = parse_args()
    selected_mode = args.mode
    verbose = args.verbose
    should_pick = args.pick and not args.no_pick

    # 1. Clean Terminal Header Banner
    print("=" * 62)
    print("           AUTONOMOUS BOOK RETRIEVAL ROBOT            ")
    print("=" * 62)

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
        choice_map = {"1": "auto", "2": "spine", "3": "scattered", "4": "mixed"}
        selected_mode = choice_map.get(choice, "auto")

    # 2. Request Book Query from User
    query_result, match_score, source = get_user_query(args.query)

    if not query_result or not query_result.get("title"):
        print("\nNo book query entered. Exiting.\n")
        return None

    query_title = query_result["title"]
    book_id = query_result.get("book_id", "")
    author = query_result.get("author", "")

    # Display clean catalog confirmation
    if book_id and book_id != "CUSTOM":
        author_str = f" by {author}" if author else ""
        print(f"\nTarget Confirmed: \"{query_title}\"{author_str} [ID: {book_id}]", flush=True)
    else:
        print(f"\nTarget Query: \"{query_title}\"", flush=True)

    # 3. Connect to RealSense Camera
    from input.camera import RealSenseCamera
    camera = RealSenseCamera()

    print("\n[1/3] Connecting to RealSense camera & analyzing scene...", flush=True)
    try:
        camera.start(verbose=verbose)
    except Exception as e:
        print(f"\n[ERROR] Could not start RealSense camera: {e}")
        print("        Please ensure the Intel RealSense D435i is connected to a USB 3.0 port.\n")
        return None

    cv_image = None
    target = None

    try:
        cv_image, depth_frame, color_intrinsics = camera.capture()
        if cv_image is None:
            print("[ERROR] Camera Capture Failed: Could not acquire RGB+Depth frame from camera.\n", flush=True)
            return None

        # Mode determination
        if selected_mode == "auto":
            from shared.mode_classifier import classify_mode
            mode, mode_conf = classify_mode(cv_image, verbose=verbose)
            mode_label = {
                "spine": "Shelf Spine (Vertical Books)",
                "scattered": "Scattered Surface (Tabletop Books)",
                "mixed": "Mixed Scene (Shelf + Desk Books)"
            }.get(mode.lower(), mode.upper())
            print(f"      Scene Arrangement: {mode_label} ({mode_conf*100:.0f}% confidence)", flush=True)
        else:
            mode = selected_mode
            print(f"      Scene Arrangement: {mode.upper()} (User Specified)", flush=True)

        # 4. Target Localization, OCR & Sub-Pixel 3D Pose Extraction
        print(f"[2/3] Localizing '{query_title}' & calculating 3D grasp coordinates...", flush=True)
        if mode == "spine":
            from spine_pipeline.find_target_spine import find_target_spine_live
            target = find_target_spine_live(
                query_title,
                camera=camera,
                cv_image=cv_image,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                refine_sam=True,
                save_visual=True
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
                save_visual=True
            )
        else:
            from mixed_pipeline.find_target_mixed import find_target_mixed_live
            target = find_target_mixed_live(
                query_title,
                camera=camera,
                cv_image=cv_image,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                refine_sam=True,
                save_visual=True
            )

    finally:
        camera.stop()

    if target is None:
        print(f"\n[NOT FOUND] No book matching '{query_title}' was localized.", flush=True)
        print("            Similarity score was below threshold (0.35). Please verify the book is in view.\n", flush=True)
        return None

    # 5. Coordinate Transform & Spatial Telemetry
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

    cx_m, cy_m, cz_m = float(pick_xyz_m[0]), float(pick_xyz_m[1]), float(pick_xyz_m[2])
    sim_score = float(target.get("similarity", 1.0)) * 100.0
    detected_title = target.get("text", query_title)

    print(f"\n" + "=" * 62)
    print(f"                    TARGET LOCALIZED                      ")
    print(f"=" * 62)
    print(f"  Title:            {detected_title}")
    print(f"  OCR Match:        {sim_score:.0f}%")
    print(f"  Camera 3D (m):    X = {cx_m:+.3f},  Y = {cy_m:+.3f},  Z = {cz_m:.3f}")
    print(f"  Dobot 3D (mm):    X = {rx:+.1f},  Y = {ry:+.1f},  Z = {rz:.1f}")
    print(f"  Grasp Angle:      R = {r_angle:+.1f}° ({'Parallel to shelf spine' if is_spine else 'Cover edge alignment'})")
    print(f"=" * 62 + "\n")

    # 6. Physical Dobot Execution
    if should_pick:
        print("[3/3] Executing robotic retrieval & deposit sequence...", flush=True)
        pick_result = dobot.execute_pick_sequence(
            pick_xyz_m,
            grasp_angle_deg=pick_angle,
            is_spine=is_spine
        )
        if pick_result.get("status") == "success":
            print(f"\n[SUCCESS] '{detected_title}' successfully retrieved and deposited!\n", flush=True)
        else:
            print(f"\n[NOTICE] Sequence completed: {pick_result.get('message', 'Execution finished')}\n", flush=True)
    else:
        print("[3/3] Manipulation skipped (--no-pick active). Coordinates localized successfully.\n", flush=True)

    return target


if __name__ == "__main__":
    run()
