"""
scattered_pipeline/find_target_scattered.py
----------------------------------------------------------------------------
Combines detect_scattered.py (SAM 3 + filtering) + ocr_match_scattered.py
(OCR + title matching) into the full scattered-book-finding flow.

Supports:
  - find_target_scattered_image(...) — run on a saved image file (testing)
  - find_target_scattered_live(...)  — run on a live RealSense frame,
                                        including depth deprojection to XYZ
                                        (uses input/camera.py + input/depth_utils.py)
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly or as package
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from PIL import Image

from input.query_input import get_book_query
from scattered_pipeline.detect_scattered import detect_books, CONF_THRESHOLD as DEFAULT_CONF
from scattered_pipeline.ocr_match_scattered import warp_obb_to_upright, ocr_best_match, get_reader

MIN_SIMILARITY_THRESHOLD = 0.35


def _pick_best_match(cv_image, detections, query_title, reader=None, min_similarity=MIN_SIMILARITY_THRESHOLD):
    """
    Runs OCR matching on every detection candidate, adds 'text' and 'similarity'
    fields to each candidate dict, and returns the best-scoring one.
    """
    if not detections:
        print("\n[Scattered] No candidates detected above confidence threshold.", flush=True)
        return None

    reader = reader or get_reader()
    print(f"\n[Scattered] Evaluating {len(detections)} candidate(s) with OCR matching...", flush=True)

    # Sort candidates by detection confidence descending
    sorted_detections = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    best = None

    for d in sorted_detections:
        crop = warp_obb_to_upright(cv_image, d["obb_corners"])
        score, text = ocr_best_match(crop, query_title, reader=reader, early_exit_score=0.65)
        d["similarity"] = score
        d["text"] = text
        d["ocr_score"] = score
        d["ocr_text"] = text

        safe_text = text[:40].encode("ascii", "backslashreplace").decode("ascii")
        print(f"  conf={d['conf']:.2f}  OCR='{safe_text}'  match={score:.2f}", flush=True)

        if best is None or score > best["similarity"]:
            best = d

        # Confident exact or near-exact match — avoid spending time OCRing other candidates
        if score >= 0.80:
            break

    if best is None or best["similarity"] < min_similarity:
        sim = best["similarity"] if best else 0.0
        print(f"\n[Scattered] Best match similarity ({sim:.2f}) is below threshold ({min_similarity:.2f}). No target found.", flush=True)
        return None

    print("\n---- Best Scattered Target Match ----", flush=True)
    print(f"OCR text:           '{best['text']}'", flush=True)
    print(f"Similarity score:    {best['similarity']:.2f}", flush=True)
    print(f"Center (u, v):       ({best['center'][0]:.1f}, {best['center'][1]:.1f})", flush=True)
    print(f"Width x Height:      {best['width_px']:.1f} x {best['height_px']:.1f} px", flush=True)
    print(f"OBB Angle:           {best['obb_angle']:.1f} deg", flush=True)
    print(f"Top-edge center:     ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)

    try:
        from shared.ocr_reader import _find_catalog_book
        cat_book = _find_catalog_book(query_title)
        if cat_book:
            best["title"] = cat_book.get("title", best.get("text"))
    except Exception:
        pass

    return best


def _refine_target_with_sam(cv_image, best, refine_mode="fast_pick", pick_location="spine_mid", query_title=None):
    """
    Refines target book coordinates using SAM.
    Supports two modes:
      - 'fast_pick': Uses SAM strictly on the grasp edge (middle of top edge for front-facing books,
                     or middle of the spine for flat books) to get the pickup point.
      - 'full': Segments the entire book cover, fitting a 4-corner perspective quadrilateral.
    """
    if best is None or "corners" not in best:
        return best

    target_query = query_title or best.get("text")

    # Front-facing upright book check:
    from shared.sam_refine import is_front_facing_book
    is_front = bool(is_front_facing_book(best.get("corners", best.get("obb_corners")), cv_image))
    best["is_front_facing"] = is_front
    if is_front:
        if pick_location in ("spine_mid", "auto", None):
            pick_location = "top_center"
            print("[Scattered Target] Front-facing upright book detected! Grasping middle of top edge.", flush=True)

    if refine_mode == "fast_pick":
        try:
            from shared.sam_refine import refine_pickup_point_with_sam
            print(f"\n[SAM Fast Pick] Target book matched ('{best['text']}'). Refining grasp pickup point on spine...", flush=True)
            res = refine_pickup_point_with_sam(cv_image, best["corners"], pick_location=pick_location, query_title=target_query)
            if res is not None:
                best["pick_point_2d"] = res["pick_point_2d"]
                best["top_edge_center"] = res["pick_point_2d"]
                best["pick_point_px"] = [res["pick_point_2d"][0], res["pick_point_2d"][1]]
                best["top_edge_center_u"] = res["pick_point_2d"][0]
                best["top_edge_center_v"] = res["pick_point_2d"][1]
                best["pick_edge_endpoints"] = res["pick_edge_endpoints"]
                best["pick_edge_angle_deg"] = res["pick_edge_angle_deg"]
                best["grasp_approach_angle_deg"] = res["grasp_approach_angle_deg"]
                best["is_short_edge"] = res["is_short_edge"]
                best["pick_edge_length_px"] = res["pick_edge_length_px"]
                best["obb_angle"] = res["pick_edge_angle_deg"]
                best["center"] = res["center_2d"]
                best["obb_center"] = res["center_2d"]
                best["pick_contour"] = res.get("pick_contour")
                best["refine_mode"] = "fast_pick"
                best["is_front_facing"] = bool(res.get("is_front_facing", is_front))

                edge_desc = "Middle of Top" if pick_location in ("top", "top_center", "front_facing", "top_mid") else "Spine Surface"
                print(f"[SAM Fast Pick] Pick Point [{edge_desc}]: ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f}) px | Grasp Angle: {best['pick_edge_angle_deg']:.1f}° | Approach: {best['grasp_approach_angle_deg']:.1f}°", flush=True)
        except Exception as e:
            print(f"[SAM Fast Pick] Target refinement fallback: {e}", flush=True)
        return best

    try:
        from shared.sam_refine import refine_obb_with_sam, extract_refined_pick_edge
        print(f"\n[SAM Refine] Target book matched ('{best['text']}'). Running full SAM boundary refinement...", flush=True)
        res = refine_obb_with_sam(cv_image, best["corners"], return_contour=True)
        if isinstance(res, tuple):
            refined_corners, sam_contour = res
        else:
            refined_corners, sam_contour = res, None

        if refined_corners is not None:
            best["corners"] = refined_corners
            best["obb_corners"] = refined_corners
            if sam_contour is not None:
                best["mask_contour"] = sam_contour

            edge_info = extract_refined_pick_edge(refined_corners, pick_location=pick_location, img=cv_image, query_title=target_query)
            best["pick_point_2d"] = edge_info["pick_point_2d"]
            best["top_edge_center"] = edge_info["pick_point_2d"]
            best["pick_point_px"] = [edge_info["pick_point_2d"][0], edge_info["pick_point_2d"][1]]
            best["top_edge_center_u"] = edge_info["pick_point_2d"][0]
            best["top_edge_center_v"] = edge_info["pick_point_2d"][1]
            best["pick_edge_endpoints"] = edge_info["pick_edge_endpoints"]
            best["pick_edge_angle_deg"] = edge_info["pick_edge_angle_deg"]
            best["grasp_approach_angle_deg"] = edge_info["grasp_approach_angle_deg"]
            best["is_short_edge"] = edge_info["is_short_edge"]
            best["pick_edge_length_px"] = edge_info["pick_edge_length_px"]
            best["obb_angle"] = edge_info["pick_edge_angle_deg"]
            best["center"] = edge_info["center_2d"]
            best["obb_center"] = edge_info["center_2d"]

            rect = cv2.minAreaRect(refined_corners)
            (cx, cy), (w, h), angle = rect
            best["obb_size"] = (w, h)
            best["width_px"] = float(min(w, h))
            best["height_px"] = float(max(w, h))
            best["refine_mode"] = "full"
            best["is_front_facing"] = bool(edge_info.get("is_front_facing", is_front))

            print(f"[SAM Refine] Target refined exact coordinates -> Center: ({best['center'][0]:.1f}, {best['center'][1]:.1f}), Spine Mid Pick: ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)
    except Exception as e:
        print(f"[SAM Refine] Target refinement fallback: {e}", flush=True)

    return best


def _save_visual_result(cv_image, detections, best_match, output_path, xyz=None):
    if best_match is None or cv_image is None:
        return cv_image

    from shared.display_utils import draw_aesthetic_target_overlay
    res = draw_aesthetic_target_overlay(cv_image, best_match)

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, res)
        print(f"\nSaved aesthetic visual result to: {output_path}", flush=True)

        from datetime import datetime
        runs_dir = Path(output_path).parent / "saved_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(output_path).stem
        history_path = runs_dir / f"{stem}_{timestamp}.jpg"
        cv2.imwrite(str(history_path), res)
    except Exception as e:
        print(f"Notice saving visual result: {e}", flush=True)

    return res


def find_target_scattered_image(image_path, query_title, conf_threshold=DEFAULT_CONF,
                                 save_visual=True, output_path="result_target_scattered.jpg",
                                 min_similarity=MIN_SIMILARITY_THRESHOLD, refine_mode="fast_pick",
                                 pick_location="spine_mid", refine_sam=None):
    """
    Runs scattered-book detection + OCR matching on a saved image file.
    Supports refine_mode='fast_pick' (default, targets pickup point without whole-book boundary)
    or refine_mode='full' (full-book boundary segmentation).
    Returns a target_output dict or None if nothing matched.
    """
    if refine_sam is not None:
        if not refine_sam:
            refine_mode = None
        elif refine_sam is True and refine_mode is None:
            refine_mode = "fast_pick"

    img_path = Path(image_path)
    print(f"\n[Scattered] Running on image: {img_path.name}  |  Query title: '{query_title}'  |  Refine: {refine_mode}", flush=True)

    cv_image, detections = detect_books(image_path=img_path, conf_threshold=conf_threshold)
    print(f"[Scattered] {len(detections)} candidate(s) detected after filtering.", flush=True)

    best = _pick_best_match(cv_image, detections, query_title, min_similarity=min_similarity)
    if best is None:
        if save_visual:
            _save_visual_result(cv_image, detections, None, output_path)
        return None

    from shared.sam_refine import is_front_facing_book
    best["is_front_facing"] = bool(is_front_facing_book(best.get("corners", best.get("obb_corners")), cv_image))

    if refine_mode:
        best = _refine_target_with_sam(cv_image, best, refine_mode=refine_mode, pick_location=pick_location, query_title=query_title)

    if save_visual:
        best["visual_frame"] = _save_visual_result(cv_image, detections, best, output_path)

    pick_2d = best.get("pick_point_2d", best.get("top_edge_center", best.get("center")))
    best["pick_point_2d"] = pick_2d
    best["pick_point_px"] = [pick_2d[0], pick_2d[1]]

    target_output = {
        "u": best["center"][0],
        "v": best["center"][1],
        "width_px": best["width_px"],
        "height_px": best["height_px"],
        "top_edge_center_u": pick_2d[0],
        "top_edge_center_v": pick_2d[1],
        "pick_point_2d": pick_2d,
        "pick_point_px": [pick_2d[0], pick_2d[1]],
        "visual_frame": best.get("visual_frame"),
        "pick_edge_endpoints": best.get("pick_edge_endpoints"),
        "pick_edge_angle_deg": best.get("pick_edge_angle_deg", best.get("obb_angle")),
        "grasp_approach_angle_deg": best.get("grasp_approach_angle_deg"),
        "is_short_edge": best.get("is_short_edge"),
        "pick_edge_length_px": best.get("pick_edge_length_px"),
        "obb_angle": best["obb_angle"],
        "obb_size": best["obb_size"],
        "similarity": best["similarity"],
        "conf": best["conf"],
        "text": best["text"],
        "title": best.get("title", best.get("text")),
        "corners": best.get("corners", best.get("obb_corners")),
        "obb_corners": best.get("obb_corners", best.get("corners")),
        "mask_contour": best.get("mask_contour"),
        "refine_mode": best.get("refine_mode", refine_mode),
        "is_front_facing": bool(best.get("is_front_facing", False)),
    }
    from shared.display_utils import print_target_summary
    print_target_summary(target_output)
    return target_output


# =========================================================
# LIVE MODE — RealSense capture + depth deprojection
# =========================================================

def find_target_scattered_live(query_title, conf_threshold=DEFAULT_CONF,
                                save_visual=True, output_path="result_target_scattered_live.jpg",
                                min_similarity=MIN_SIMILARITY_THRESHOLD, camera=None,
                                cv_image=None, depth_frame=None, color_intrinsics=None,
                                refine_mode="fast_pick", pick_location="spine_mid",
                                refine_sam=None):
    """
    Full live pipeline: captures an aligned color+depth frame from RealSense,
    runs trained YOLO OBB detection + OCR matching, then uses SAM to pinpoint the
    exact grasp pickup point (refine_mode='fast_pick') before deprojecting to 3D XYZ.

    Supports reusing an existing camera or pre-captured frame.
    """
    from input.camera import RealSenseCamera
    from input.depth_utils import deproject_target

    if refine_sam is not None:
        if not refine_sam:
            refine_mode = None
        elif refine_sam is True and refine_mode is None:
            refine_mode = "fast_pick"

    have_external_frame = (cv_image is not None and depth_frame is not None and color_intrinsics is not None)
    owns_camera = (camera is None and not have_external_frame)

    if owns_camera:
        camera = RealSenseCamera()
        camera.start()
    elif camera is not None and not getattr(camera, "_started", False):
        camera.start()

    try:
        if not have_external_frame:
            cv_image, depth_frame, color_intrinsics = camera.capture()
            if cv_image is None:
                print("[Scattered] Failed to capture frame from RealSense.", flush=True)
                return None

        cv_image, detections = detect_books(image=cv_image, conf_threshold=conf_threshold)
        print(f"[Scattered] {len(detections)} candidate(s) detected.", flush=True)

        best = _pick_best_match(cv_image, detections, query_title, min_similarity=min_similarity)
        if best is None:
            if save_visual:
                _save_visual_result(cv_image, detections, None, output_path)
            return None

        if refine_mode:
            best = _refine_target_with_sam(cv_image, best, refine_mode=refine_mode, pick_location=pick_location, query_title=query_title)

        u, v = best["center"]
        pick_u, pick_v = best["top_edge_center"]
        xyz_center, xyz_pick = None, None
        if depth_frame is not None and color_intrinsics is not None:
            from input.depth_utils import extract_robust_book_depth
            book_depth = extract_robust_book_depth(
                depth_frame,
                corners=best.get("corners", best.get("obb_corners")),
                center=(u, v),
                mask_contour=best.get("mask_contour")
            )
            xyz_center = deproject_target(
                depth_frame, color_intrinsics, u, v,
                reference_depth=book_depth,
                corners=best.get("corners")
            )
            xyz_pick = deproject_target(
                depth_frame, color_intrinsics, pick_u, pick_v,
                reference_depth=book_depth,
                corners=best.get("corners")
            )

            if (xyz_pick is None or xyz_pick[2] > 0.65) and book_depth is not None:
                print(f"[Scattered] Pick depth was background (>0.65m) or missing. Using robust book depth ({book_depth:.3f}m).", flush=True)
                xyz_pick = deproject_target(
                    depth_frame, color_intrinsics, pick_u, pick_v,
                    reference_depth=book_depth
                )

        if xyz_pick is None and xyz_center is not None:
            xyz_pick = xyz_center
        if xyz_center is None and xyz_pick is not None:
            xyz_center = xyz_pick

        # Kinematic Reachability Validation & Fallback for Scattered Books:
        # If the nominal pick point is out of reach of the Dobot arm, search the safe
        # interior of the book cover for a pick position that can still get the book.
        is_fallback_reachable = False
        try:
            from shared.reachable_pick import find_reachable_scattered_point
            reach_info = find_reachable_scattered_point(
                corners_2d=best.get("corners", best.get("obb_corners")),
                nominal_u=pick_u,
                nominal_v=pick_v,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                book_depth=book_depth,
                nominal_xyz_cam=xyz_pick
            )
            if reach_info is not None and reach_info.get("is_fallback"):
                xyz_pick = reach_info["pick_xyz_m"]
                pick_u, pick_v = reach_info["pick_uv"]
                best["pick_point_2d"] = (pick_u, pick_v)
                best["top_edge_center"] = (pick_u, pick_v)
                best["robot_xyz_mm"] = reach_info["pick_rob_mm"]
                best["pick_note"] = reach_info["reason"]
                is_fallback_reachable = True
        except Exception as e:
            print(f"[Scattered Reachability] Notice during reachability check: {e}", flush=True)

        best["xyz_center_camera_frame"] = xyz_center
        best["xyz_pick_camera_frame"] = xyz_pick
        best_xyz = xyz_pick if xyz_pick is not None else xyz_center
        best["xyz_camera_frame"] = best_xyz
        best["is_fallback_reachable"] = is_fallback_reachable

        best["pick_point_px"] = [pick_u, pick_v]
        if save_visual:
            best["visual_frame"] = _save_visual_result(cv_image, detections, best, output_path, xyz=best_xyz)

        target_output = {
            "u": u,
            "v": v,
            "width_px": best["width_px"],
            "height_px": best["height_px"],
            "top_edge_center_u": pick_u,
            "top_edge_center_v": pick_v,
            "pick_point_2d": best.get("pick_point_2d", (pick_u, pick_v)),
            "pick_point_px": [pick_u, pick_v],
            "visual_frame": best.get("visual_frame"),
            "is_fallback_reachable": best.get("is_fallback_reachable", False),
            "robot_xyz": best.get("robot_xyz_mm"),
            "pick_note": best.get("pick_note"),
            "pick_edge_endpoints": best.get("pick_edge_endpoints"),
            "pick_edge_angle_deg": best.get("pick_edge_angle_deg", best.get("obb_angle")),
            "grasp_approach_angle_deg": best.get("grasp_approach_angle_deg"),
            "is_short_edge": best.get("is_short_edge"),
            "pick_edge_length_px": best.get("pick_edge_length_px"),
            "obb_angle": best["obb_angle"],
            "obb_size": best["obb_size"],
            "similarity": best["similarity"],
            "conf": best["conf"],
            "text": best["text"],
        "title": best.get("title", best.get("text")),
            "xyz_camera_frame": best_xyz,
            "xyz_pick_camera_frame": xyz_pick,
            "xyz_center_camera_frame": xyz_center,
            "corners": best.get("corners", best.get("obb_corners")),
            "obb_corners": best.get("obb_corners", best.get("corners")),
            "mask_contour": best.get("mask_contour"),
            "refine_mode": best.get("refine_mode", refine_mode),
            "is_front_facing": bool(best.get("is_front_facing", False)),
        }
        from shared.display_utils import print_target_summary
        print_target_summary(target_output)
        return target_output

    except Exception as e:
        print(f"\n[Scattered] RealSense Camera error: {e}", flush=True)
        return None

    finally:
        if owns_camera and camera is not None:
            camera.stop()


# Alias for compatibility
find_target_book = find_target_scattered_image


if __name__ == "__main__":
    print("=== Step 1: Reader Input Query ===", flush=True)
    query_result = get_book_query()

    if query_result:
        target_title = query_result["title"]
        print(f"\n=== Step 2: Finding Scattered Target for '{target_title}' ===", flush=True)
        find_target_scattered_live(target_title)
    else:
        print("\n[Scattered] No valid query received.")
