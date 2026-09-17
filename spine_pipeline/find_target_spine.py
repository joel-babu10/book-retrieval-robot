"""
spine_pipeline/find_target_spine.py
----------------------------------------------------------------------------
Combines detect_spine.py (YOLO OBB) + ocr_match_spine.py (OCR + title
matching) into the full spine-finding flow. Supports:

  - find_target_spine_image(...)  — run on a saved image file (testing)
  - find_target_spine_live(...)   — run on a live RealSense frame,
                                     including depth deprojection to XYZ
                                     (uses input/camera.py + input/depth_utils.py)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly or as package
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from input.query_input import get_book_query
from spine_pipeline.detect_spine import detect_spines, CONF_THRESHOLD as DEFAULT_CONF
from spine_pipeline.ocr_match_spine import ocr_spine_crop, get_reader


def _pick_best_match(img, detections, target_title, reader=None, min_similarity=0.35):
    """Runs OCR matching on every detection, returns the best-scoring one
    (with 'text' and 'similarity' added), or None if there were no
    detections or if best match is below min_similarity."""
    if not detections:
        print("\n[Spine] No detections above threshold.", flush=True)
        return None

    reader = reader or get_reader()

    for d in detections:
        text, similarity = ocr_spine_crop(img, d["corners"], target_title, reader=reader)
        d["text"] = text
        d["similarity"] = similarity
        print(f"  conf={d['conf']:.2f}  OCR='{text}'  match={similarity:.2f}", flush=True)

    best = max(detections, key=lambda d: d["similarity"])

    if best["similarity"] < min_similarity:
        print(f"\n[Spine] Best match similarity ({best['similarity']:.2f}) is below threshold ({min_similarity:.2f}). No confident target found.", flush=True)
        return None

    print("\n---- Best Match ----", flush=True)
    print(f"OCR text:           '{best['text']}'", flush=True)
    print(f"Similarity score:    {best['similarity']:.2f}", flush=True)
    print(f"Center (u, v):       ({best['center'][0]:.1f}, {best['center'][1]:.1f})", flush=True)
    print(f"Width:               {best['width_px']:.1f}px", flush=True)
    print(f"Top-edge center:     ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)

    return best


def _save_visual_result(img, best, output_path, xyz=None):
    pts = best["corners"].astype(int).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    img_h, img_w = img.shape[:2]

    def clamp_point(x, y, w=0, h=0, pad=5):
        x = max(pad, min(x, img_w - w - pad))
        y = max(pad + h, min(y, img_h - pad))
        return int(x), int(y)

    u, v = best["top_edge_center"]
    cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)
    lx, ly = clamp_point(u + 10, v - 10, w=90, h=15)
    cv2.putText(img, f"({u:.0f},{v:.0f})", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cx, cy = best["center"]
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0), markerType=cv2.MARKER_CROSS,
                   markerSize=14, thickness=2)
    lx2, ly2 = clamp_point(cx + 10, cy + 20, w=90, h=15)
    cv2.putText(img, f"({cx:.0f},{cy:.0f})", (lx2, ly2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    info_lines = [
        f"OCR: {best['text'][:30]}",
        f"Match: {best['similarity']:.2f}",
        f"Conf: {best['conf']:.2f}",
        f"Center: ({cx:.1f}, {cy:.1f})",
        f"Width: {best['width_px']:.1f}px",
        f"Top-edge center: ({u:.1f}, {v:.1f})",
    ]
    if xyz is not None:
        x, y, z = xyz
        info_lines.append(f"XYZ: ({x:.3f}, {y:.3f}, {z:.3f}) m")

    text_x = int(cx) + 15
    box_w, box_h = 260, 22 * len(info_lines) + 10
    text_y = int(cy) - 60

    text_x = max(5, min(text_x, img_w - box_w - 5))
    text_y = max(30, min(text_y, img_h - box_h - 5))

    overlay = img.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - 20), (text_x + box_w, text_y + box_h - 15), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    for i, line in enumerate(info_lines):
        y = text_y + i * 22
        cv2.putText(img, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"\nSaved visual result to: {output_path}", flush=True)


# =========================================================
# IMAGE MODE — test on a saved file, no camera/depth needed
# =========================================================

def find_target_spine_image(image_path, target_title, conf_threshold=DEFAULT_CONF,
                             save_visual=True, output_path="result_target_spine.jpg"):
    """
    Runs spine detection + OCR matching on a saved image file.
    Returns a target_output dict (u, v, width_px, top-edge center, etc.)
    or None if nothing matched.
    """
    img, detections = detect_spines(image_path=image_path, conf_threshold=conf_threshold)
    print(f"[Spine] {len(detections)} spine candidate(s) detected.", flush=True)

    best = _pick_best_match(img, detections, target_title)
    if best is None:
        return None

    target_output = {
        "u": best["center"][0],
        "v": best["center"][1],
        "width_px": best["width_px"],
        "top_edge_center_u": best["top_edge_center"][0],
        "top_edge_center_v": best["top_edge_center"][1],
        "similarity": best["similarity"],
        "conf": best["conf"],
        "text": best["text"],
    }
    print("\n---- Target Output ----", flush=True)
    print(target_output, flush=True)

    if save_visual:
        _save_visual_result(img, best, output_path)

    return target_output


# =========================================================
# LIVE MODE — RealSense capture + depth deprojection
# =========================================================

def find_target_spine_live(target_title, conf_threshold=DEFAULT_CONF,
                            save_visual=True, output_path="result_target_spine_live.jpg"):
    """
    Full live pipeline: captures an aligned color+depth frame from the
    RealSense camera, runs spine detection + OCR matching, then deprojects
    the matched spine's centroid to real-world XYZ (camera optical frame —
    still needs the Step 6 transform for the arm's base frame).
    """
    from input.camera import RealSenseCamera
    from input.depth_utils import deproject_target

    try:
        with RealSenseCamera() as cam:
            cv_image, depth_frame, color_intrinsics = cam.capture()
            if cv_image is None:
                print("[Spine] Failed to capture frame from RealSense.", flush=True)
                return None

            img, detections = detect_spines(image=cv_image, conf_threshold=conf_threshold)
            print(f"[Spine] {len(detections)} spine candidate(s) detected.", flush=True)

            best = _pick_best_match(img, detections, target_title)
            if best is None:
                return None

            u, v = best["center"]
            xyz = deproject_target(depth_frame, color_intrinsics, u, v)

            if save_visual:
                _save_visual_result(img.copy(), best, output_path, xyz=xyz)

            target_output = {
                "u": u,
                "v": v,
                "width_px": best["width_px"],
                "top_edge_center_u": best["top_edge_center"][0],
                "top_edge_center_v": best["top_edge_center"][1],
                "similarity": best["similarity"],
                "conf": best["conf"],
                "text": best["text"],
                "xyz_camera_frame": xyz,
            }
            print("\n---- Target Output ----", flush=True)
            print(target_output, flush=True)

            return target_output

    except Exception as e:
        print(f"\n[Spine] RealSense Camera error: {e}", flush=True)
        return None


if __name__ == "__main__":
    print("=== Step 1: Reader Input Query ===", flush=True)
    query_result = get_book_query()

    if query_result:
        target_title = query_result["title"]
        print(f"\n=== Step 2: Finding Target Spine for '{target_title}' ===", flush=True)
        find_target_spine_live(target_title)
    else:
        print("\n[Spine] No valid query received.")
