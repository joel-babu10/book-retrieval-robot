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

    return best


def _refine_target_with_sam(cv_image, best, refine_mode="fast_pick", pick_location="top_left"):
    """
    Refines target book coordinates using SAM.
    Supports two modes:
      - 'fast_pick': Uses SAM strictly on the grasp edge / top-left corner to get the
                     pickup point without segmenting or bounding the entire book.
      - 'full': Segments the entire book cover, fitting a 4-corner perspective quadrilateral.
    """
    if best is None or "corners" not in best:
        return best

    if refine_mode == "fast_pick":
        try:
            from shared.sam_refine import refine_pickup_point_with_sam
            print(f"\n[SAM Fast Pick] Target book matched ('{best['text']}'). Refining grasp pickup point...", flush=True)
            res = refine_pickup_point_with_sam(cv_image, best["corners"], pick_location=pick_location)
            if res is not None:
                best["pick_point_2d"] = res["pick_point_2d"]
                best["top_edge_center"] = res["pick_point_2d"]
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

                print(f"[SAM Fast Pick] Pick Point: ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f}) px | Grasp Angle: {best['pick_edge_angle_deg']:.1f}° | Approach: {best['grasp_approach_angle_deg']:.1f}°", flush=True)
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

            edge_info = extract_refined_pick_edge(refined_corners, pick_location=pick_location)
            best["pick_point_2d"] = edge_info["pick_point_2d"]
            best["top_edge_center"] = edge_info["pick_point_2d"]
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

            print(f"[SAM Refine] Target refined exact coordinates -> Center: ({best['center'][0]:.1f}, {best['center'][1]:.1f}), Top-Left Pick: ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)
    except Exception as e:
        print(f"[SAM Refine] Target refinement fallback: {e}", flush=True)

    return best


def _save_visual_result(cv_image, detections, best_match, output_path, xyz=None):
    """
    Renders visual results cleanly on cv_image and saves to output_path.
    Adapts based on refine_mode:
      - 'fast_pick': Renders ONLY the precision pick target, grasp edge alignment line,
                     and approach normal vector (no whole-book boundary clutter).
      - 'full': Renders full translucent SAM mask shading, yellow organic contour, and green quad box.
    """
    img = cv_image.copy()
    img_h, img_w = img.shape[:2]

    if best_match is not None:
        refine_mode = best_match.get("refine_mode", "full")
        cx, cy = best_match["center"]
        u_top, v_top = best_match["top_edge_center"]
        pick_angle = best_match.get("pick_edge_angle_deg", best_match.get("obb_angle", 0.0))
        approach_angle = best_match.get("grasp_approach_angle_deg", (pick_angle + 90.0))

        if refine_mode == "fast_pick":
            # FAST PICK MODE: Render only the pickup point and grasp vector without whole-book boundary
            pick_cnt = best_match.get("pick_contour")
            if pick_cnt is not None and len(pick_cnt) >= 3:
                cv2.drawContours(img, [pick_cnt.astype(np.int32)], -1, (0, 255, 255), 2, cv2.LINE_AA)

            endpoints = best_match.get("pick_edge_endpoints")
            if endpoints and len(endpoints) == 2:
                p1 = (int(endpoints[0][0]), int(endpoints[0][1]))
                p2 = (int(endpoints[1][0]), int(endpoints[1][1]))
                cv2.line(img, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)  # Cyan grasp edge
                cv2.circle(img, p1, 5, (255, 255, 0), -1)
                cv2.circle(img, p2, 5, (255, 255, 0), -1)

            # Draw approach normal vector (green arrow showing arm entry direction)
            norm_rad = np.radians(approach_angle)
            nx = 50 * np.cos(norm_rad)
            ny = 50 * np.sin(norm_rad)
            cv2.arrowedLine(img, (int(u_top - nx), int(v_top - ny)), (int(u_top), int(v_top)),
                            (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)

            # Precision Pick Point Marker (red circle with white cross)
            cv2.circle(img, (int(u_top), int(v_top)), 8, (0, 0, 255), -1)
            cv2.drawMarker(img, (int(u_top), int(v_top)), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            cv2.putText(img, "PICK [TOP-LEFT]", (int(u_top) - 25, max(25, int(v_top) - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

            title_header = "SAM FAST-PICK TARGET [SCATTERED]"

        else:
            # FULL BOUNDARY MODE: Translucent SAM Mask Shading & Organic Contour
            sam_cnt = best_match.get("mask_contour")
            if sam_cnt is not None and len(sam_cnt) >= 3:
                mask_overlay = img.copy()
                cv2.fillPoly(mask_overlay, [sam_cnt.astype(np.int32)], (0, 242, 254))
                img = cv2.addWeighted(mask_overlay, 0.22, img, 0.78, 0)
                cv2.drawContours(img, [sam_cnt.astype(np.int32)], -1, (0, 255, 255), 2, cv2.LINE_AA)

            pts = best_match["obb_corners"].astype(int).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

            endpoints = best_match.get("pick_edge_endpoints")
            if endpoints and len(endpoints) == 2:
                p1 = (int(endpoints[0][0]), int(endpoints[0][1]))
                p2 = (int(endpoints[1][0]), int(endpoints[1][1]))
                cv2.line(img, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(img, p1, 5, (255, 255, 0), -1)
                cv2.circle(img, p2, 5, (255, 255, 0), -1)

            cv2.circle(img, (int(u_top), int(v_top)), 7, (0, 0, 255), -1)
            cv2.drawMarker(img, (int(u_top), int(v_top)), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(img, "PICK [TOP-LEFT]", (int(u_top) - 20, max(20, int(v_top) - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            title_header = "SAM-REFINED TARGET [SCATTERED]"

        info_lines = [
            title_header,
            f"OCR: {best_match['text'][:28]}",
            f"Match: {best_match['similarity']:.2f}",
            f"Conf: {best_match['conf']:.2f}",
            f"Top-Left Pick: ({u_top:.1f}, {v_top:.1f})",
            f"Grasp Angle: {pick_angle:.1f} deg",
            f"Approach Angle: {approach_angle:.1f} deg",
        ]

        pick_xyz = best_match.get("xyz_pick_camera_frame") or xyz
        if pick_xyz is not None:
            info_lines.append(f"Pick XYZ: ({pick_xyz[0]:.3f}, {pick_xyz[1]:.3f}, {pick_xyz[2]:.3f}) m")

        # Position HUD overlay cleanly in top margin
        text_x = 35
        text_y = 45
        box_w, box_h = 285, 20 * len(info_lines) + 15

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - 25), (text_x + box_w, text_y + box_h - 20), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

        for i, line in enumerate(info_lines):
            y_pos = text_y + i * 20
            color = (0, 255, 255) if i == 0 else (0, 255, 0)
            cv2.putText(img, line, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Header tag over the target book
        cv2.putText(img, f"TARGET: {best_match['text'][:20]}", (int(cx) - 45, max(25, int(v_top) - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"\nSaved visual result to: {output_path}", flush=True)

    try:
        from datetime import datetime
        runs_dir = Path(output_path).parent / "saved_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(output_path).stem
        history_path = runs_dir / f"{stem}_{timestamp}.jpg"
        cv2.imwrite(str(history_path), img)
        print(f"Stored image copy to: {history_path}", flush=True)
    except Exception as e:
        print(f"Warning storing image history: {e}", flush=True)


# =========================================================
# IMAGE MODE — test on a saved file, no camera/depth needed
# =========================================================

def find_target_scattered_image(image_path, query_title, conf_threshold=DEFAULT_CONF,
                                 save_visual=True, output_path="result_target_scattered.jpg",
                                 min_similarity=MIN_SIMILARITY_THRESHOLD, refine_mode="fast_pick",
                                 pick_location="top_left", refine_sam=None):
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

    if refine_mode:
        best = _refine_target_with_sam(cv_image, best, refine_mode=refine_mode, pick_location=pick_location)

    if save_visual:
        _save_visual_result(cv_image, detections, best, output_path)

    target_output = {
        "u": best["center"][0],
        "v": best["center"][1],
        "width_px": best["width_px"],
        "height_px": best["height_px"],
        "top_edge_center_u": best["top_edge_center"][0],
        "top_edge_center_v": best["top_edge_center"][1],
        "pick_point_2d": best.get("pick_point_2d", best["top_edge_center"]),
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
        "corners": best.get("corners", best.get("obb_corners")),
        "obb_corners": best.get("obb_corners", best.get("corners")),
        "mask_contour": best.get("mask_contour"),
        "refine_mode": best.get("refine_mode", refine_mode),
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
                                refine_mode="fast_pick", pick_location="top_left",
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
            best = _refine_target_with_sam(cv_image, best, refine_mode=refine_mode, pick_location=pick_location)

        u, v = best["center"]
        pick_u, pick_v = best["top_edge_center"]
        xyz_center, xyz_pick = None, None
        if depth_frame is not None and color_intrinsics is not None:
            xyz_center = deproject_target(depth_frame, color_intrinsics, u, v)
            xyz_pick = deproject_target(depth_frame, color_intrinsics, pick_u, pick_v)

        best["xyz_center_camera_frame"] = xyz_center
        best["xyz_pick_camera_frame"] = xyz_pick
        best["xyz_camera_frame"] = xyz_pick or xyz_center

        if save_visual:
            _save_visual_result(cv_image, detections, best, output_path, xyz=xyz_pick or xyz_center)

        target_output = {
            "u": u,
            "v": v,
            "width_px": best["width_px"],
            "height_px": best["height_px"],
            "top_edge_center_u": pick_u,
            "top_edge_center_v": pick_v,
            "pick_point_2d": best.get("pick_point_2d", (pick_u, pick_v)),
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
            "xyz_camera_frame": xyz_pick or xyz_center,
            "xyz_pick_camera_frame": xyz_pick,
            "xyz_center_camera_frame": xyz_center,
            "corners": best.get("corners", best.get("obb_corners")),
            "obb_corners": best.get("obb_corners", best.get("corners")),
            "mask_contour": best.get("mask_contour"),
            "refine_mode": best.get("refine_mode", refine_mode),
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
