from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from difflib import SequenceMatcher

# ---- CONFIG ----
MODEL_PATH = r"C:\Users\joelv\runs\obb\train-3\weights\best.pt"
CONF_THRESHOLD = 0.5
OCR_LANGUAGES = ['en']   # dropped 'ko' - mixed-language OCR was hurting Latin spine text accuracy

# ---- LOAD MODEL & OCR ----
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(OCR_LANGUAGES)


def warp_obb_to_upright(img, corners):
    """
    Takes 4 corner points of a rotated box and returns a straightened
    (de-rotated) crop of just that spine, upright and rectangular.
    """
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_pts = sorted_by_y[:2]
    bottom_pts = sorted_by_y[2:]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    tl, tr = top_pts
    bl, br = bottom_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = max(int(max(width_top, width_bottom)), 1)
    out_h = max(int(max(height_left, height_right)), 1)

    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped


def find_target(image_path, target_title, conf_threshold=CONF_THRESHOLD, save_visual=True,
                 output_path="result_target_book_obb.jpg"):
    """
    Runs OBB detection + OCR on image_path, finds the spine whose OCR text
    best matches target_title, and returns its localization data.
    Returns None if no detections pass conf_threshold.
    """
    results = model(image_path, imgsz=640)[0]
    img = cv2.imread(image_path)

    candidates = []

    for obb in results.obb:
        conf = obb.conf[0].item()
        if conf < conf_threshold:
            continue

        corners = obb.xyxyxyxy[0].cpu().numpy()
        xywhr = obb.xywhr[0].cpu().numpy()

        crop = warp_obb_to_upright(img, corners)

        # spine text usually runs top-to-bottom -> rotate for OCR, but the
        # correct direction varies by spine, so try both and keep the winner
        if crop.shape[0] > crop.shape[1]:  # taller than wide = vertical spine
            crop_cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            crop_ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            text_cw = " ".join(reader.readtext(crop_cw, detail=0)).strip()
            text_ccw = " ".join(reader.readtext(crop_ccw, detail=0)).strip()

            ratio_cw = SequenceMatcher(None, text_cw.lower(), target_title.lower()).ratio()
            ratio_ccw = SequenceMatcher(None, text_ccw.lower(), target_title.lower()).ratio()

            text = text_cw if ratio_cw >= ratio_ccw else text_ccw
        else:
            text = " ".join(reader.readtext(crop, detail=0)).strip()

        # Full-string ratio (penalizes short targets against long OCR text)
        full_ratio = SequenceMatcher(None, text.lower(), target_title.lower()).ratio()

        # Substring/partial match: check best-matching window of OCR text
        # against the target length, so short titles aren't unfairly penalized
        target_lower = target_title.lower()
        text_lower = text.lower()
        if target_lower in text_lower:
            partial_ratio = 1.0
        else:
            best_partial = 0.0
            window = len(target_lower)
            for i in range(max(1, len(text_lower) - window + 1)):
                chunk = text_lower[i:i + window]
                r = SequenceMatcher(None, chunk, target_lower).ratio()
                best_partial = max(best_partial, r)
            partial_ratio = best_partial

        similarity = max(full_ratio, partial_ratio)

        center_u = float(np.mean(corners[:, 0]))
        center_v = float(np.mean(corners[:, 1]))
        spine_width_px = float(min(xywhr[2], xywhr[3]))
        sorted_by_y2 = corners[np.argsort(corners[:, 1])]
        top_two = sorted_by_y2[:2]
        top_edge_center_u = float(np.mean(top_two[:, 0]))
        top_edge_center_v = float(np.mean(top_two[:, 1]))

        candidates.append({
            "corners": corners,
            "conf": conf,
            "text": text,
            "similarity": similarity,
            "width": spine_width_px,
            "top_edge_center": (top_edge_center_u, top_edge_center_v),
            "center": (center_u, center_v),
        })

        print(f"conf={conf:.2f}  OCR='{text}'  match={similarity:.2f}")

    if not candidates:
        print("\nNo detections above threshold.")
        return None

    # ---- PICK BEST MATCH ----
    best = max(candidates, key=lambda c: c["similarity"])

    print("\n---- Best Match ----")
    print(f"OCR text:           '{best['text']}'")
    print(f"Similarity score:    {best['similarity']:.2f}")
    print(f"Center (u, v):       ({best['center'][0]:.1f}, {best['center'][1]:.1f})")
    print(f"Width:               {best['width']:.1f}px")
    print(f"Top-edge center:     ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})")

    target_output = {
        "u": best["center"][0],
        "v": best["center"][1],
        "width_px": best["width"],
        "top_edge_center_u": best["top_edge_center"][0],
        "top_edge_center_v": best["top_edge_center"][1],
        "similarity": best["similarity"],
        "conf": best["conf"],
        "text": best["text"],
    }
    print("\n---- Target Output ----")
    print(target_output)

    if save_visual:
        _save_visual_result(img, best, output_path)

    return target_output


def _save_visual_result(img, best, output_path):
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
        f"Width: {best['width']:.1f}px",
        f"Top-edge center: ({u:.1f}, {v:.1f})",
    ]

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
    print(f"\nSaved visual result to: {output_path}")


if __name__ == "__main__":
    IMAGE_PATH = r"test\images\test_img.png"
    TARGET_TITLE = "the odessey"   # <-- change this
    find_target(IMAGE_PATH, TARGET_TITLE)