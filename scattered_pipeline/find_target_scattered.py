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

    for d in detections:
        crop = warp_obb_to_upright(cv_image, d["obb_corners"])
        score, text = ocr_best_match(crop, query_title, reader=reader)
        d["similarity"] = score
        d["text"] = text
        d["ocr_score"] = score
        d["ocr_text"] = text

        safe_text = text[:40].encode("ascii", "backslashreplace").decode("ascii")
        print(f"  conf={d['conf']:.2f}  OCR='{safe_text}'  match={score:.2f}", flush=True)

    best = max(detections, key=lambda d: d["similarity"])

    if best["similarity"] < min_similarity:
        print(f"\n[Scattered] Best match similarity ({best['similarity']:.2f}) is below threshold ({min_similarity:.2f}). No target found.", flush=True)
        return None

    print("\n---- Best Scattered Target Match ----", flush=True)
    print(f"OCR text:           '{best['text']}'", flush=True)
    print(f"Similarity score:    {best['similarity']:.2f}", flush=True)
    print(f"Center (u, v):       ({best['center'][0]:.1f}, {best['center'][1]:.1f})", flush=True)
    print(f"Width x Height:      {best['width_px']:.1f} x {best['height_px']:.1f} px", flush=True)
    print(f"OBB Angle:           {best['obb_angle']:.1f} deg", flush=True)
    print(f"Top-edge center:     ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)

    return best


def _save_visual_result(cv_image, detections, best_match, output_path, xyz=None):
    """
    Renders mask contours, OBB polylines, centroids, and labels on cv_image
    and saves to output_path.
    """
    img = cv_image.copy()
    img_h, img_w = img.shape[:2]

    for d in detections:
        is_target = (best_match is not None and d is best_match)
        color_mask = (0, 255, 0) if is_target else (150, 150, 150)
        color_box = (0, 0, 255) if is_target else (150, 150, 150)
        thickness = 3 if is_target else 1

        cv2.drawContours(img, [d["mask_contour"]], -1, color_mask, thickness)
        obb_pts = d["obb_corners"].astype(int).reshape((-1, 1, 2))
        cv2.polylines(img, [obb_pts], isClosed=True, color=color_box, thickness=thickness)

        cx, cy = d["center"]
        cv2.drawMarker(img, (int(cx), int(cy)),
                       (255, 0, 0) if is_target else (100, 100, 100),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        if not is_target:
            label = f"conf={d['conf']:.2f} ocr={d.get('similarity', 0):.2f}"
            cv2.putText(img, label, (int(cx) - 40, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    if best_match is not None:
        cx, cy = best_match["center"]
        u_top, v_top = best_match["top_edge_center"]

        # Highlight top edge center point
        cv2.circle(img, (int(u_top), int(v_top)), 6, (0, 0, 255), -1)

        info_lines = [
            f"TARGET MATCH",
            f"OCR: {best_match['text'][:30]}",
            f"Match: {best_match['similarity']:.2f}",
            f"Conf: {best_match['conf']:.2f}",
            f"Center: ({cx:.1f}, {cy:.1f})",
            f"Size: {best_match['width_px']:.0f}x{best_match['height_px']:.0f} px",
            f"Angle: {best_match['obb_angle']:.1f} deg",
        ]
        if xyz is not None:
            x, y, z = xyz
            info_lines.append(f"XYZ: ({x:.3f}, {y:.3f}, {z:.3f}) m")

        text_x = int(cx) + 15
        box_w, box_h = 260, 20 * len(info_lines) + 10
        text_y = int(cy) - 60

        text_x = max(5, min(text_x, img_w - box_w - 5))
        text_y = max(30, min(text_y, img_h - box_h - 5))

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 5, text_y - 20), (text_x + box_w, text_y + box_h - 15), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

        for i, line in enumerate(info_lines):
            y_pos = text_y + i * 20
            color = (0, 255, 255) if i == 0 else (0, 255, 0)
            cv2.putText(img, line, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"\nSaved visual result to: {output_path}", flush=True)


# =========================================================
# IMAGE MODE — test on a saved file, no camera/depth needed
# =========================================================

def find_target_scattered_image(image_path, query_title, conf_threshold=DEFAULT_CONF,
                                 save_visual=True, output_path="result_target_scattered.jpg",
                                 min_similarity=MIN_SIMILARITY_THRESHOLD):
    """
    Runs scattered-book detection + OCR matching on a saved image file.
    Returns a target_output dict or None if nothing matched.
    """
    img_path = Path(image_path)
    print(f"\n[Scattered] Running on image: {img_path.name}  |  Query title: '{query_title}'", flush=True)

    cv_image, detections = detect_books(image_path=img_path, conf_threshold=conf_threshold)
    print(f"[Scattered] {len(detections)} candidate(s) detected after filtering.", flush=True)

    best = _pick_best_match(cv_image, detections, query_title, min_similarity=min_similarity)
    if best is None:
        if save_visual:
            _save_visual_result(cv_image, detections, None, output_path)
        return None

    if save_visual:
        _save_visual_result(cv_image, detections, best, output_path)

    target_output = {
        "u": best["center"][0],
        "v": best["center"][1],
        "width_px": best["width_px"],
        "height_px": best["height_px"],
        "top_edge_center_u": best["top_edge_center"][0],
        "top_edge_center_v": best["top_edge_center"][1],
        "obb_angle": best["obb_angle"],
        "obb_size": best["obb_size"],
        "similarity": best["similarity"],
        "conf": best["conf"],
        "text": best["text"],
    }
    print("\n---- Target Output ----", flush=True)
    print(target_output, flush=True)
    return target_output


# =========================================================
# LIVE MODE — RealSense capture + depth deprojection
# =========================================================

def find_target_scattered_live(query_title, conf_threshold=DEFAULT_CONF,
                                save_visual=True, output_path="result_target_scattered_live.jpg",
                                min_similarity=MIN_SIMILARITY_THRESHOLD, camera=None,
                                cv_image=None, depth_frame=None, color_intrinsics=None):
    """
    Full live pipeline: captures an aligned color+depth frame from RealSense,
    runs SAM 3 detection + OCR matching, then deprojects the target's centroid
    to real-world XYZ (camera optical frame).

    Supports reusing an existing camera or pre-captured frame.
    """
    from input.camera import RealSenseCamera
    from input.depth_utils import deproject_target

    have_external_frame = (cv_image is not None and depth_frame is not None and color_intrinsics is not None)
    owns_camera = (camera is None and not have_external_frame)

    if owns_camera:
        camera = RealSenseCamera()
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

        u, v = best["center"]
        xyz = deproject_target(depth_frame, color_intrinsics, u, v)

        if save_visual:
            _save_visual_result(cv_image, detections, best, output_path, xyz=xyz)

        target_output = {
            "u": u,
            "v": v,
            "width_px": best["width_px"],
            "height_px": best["height_px"],
            "top_edge_center_u": best["top_edge_center"][0],
            "top_edge_center_v": best["top_edge_center"][1],
            "obb_angle": best["obb_angle"],
            "obb_size": best["obb_size"],
            "similarity": best["similarity"],
            "conf": best["conf"],
            "text": best["text"],
            "xyz_camera_frame": xyz,
        }
        print("\n---- Target Output ----", flush=True)
        print(target_output, flush=True)
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
