"""
spine_pipeline/ocr_match_spine.py
----------------------------------------------------------------------------
OCR + title-matching logic for detected spines. Kept separate from
detection so OCR tuning (rotation handling, similarity scoring) doesn't
touch the YOLO detection code.
"""

import cv2
import numpy as np
import torch
import easyocr
from difflib import SequenceMatcher

OCR_LANGUAGES = ['en', 'ch_sim']
USE_GPU = torch.cuda.is_available()

_reader = None


def get_reader(languages=OCR_LANGUAGES, gpu=USE_GPU):
    """Lazy-load EasyOCR once, reused across calls."""
    global _reader
    if _reader is None:
        print(f"[Spine OCR] Loading EasyOCR ({languages}, gpu={gpu})", flush=True)
        _reader = easyocr.Reader(languages, gpu=gpu)
    return _reader


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
    return cv2.warpPerspective(img, M, (out_w, out_h))


def title_similarity(text, target_title):
    """
    Robust title similarity matching supporting English & Chinese:
    1. Exact substring match
    2. Word token set containment
    3. Sliding window substring ratio (prevents long cover text from diluting title score)
    4. Full-string ratio
    """
    if not text or not target_title:
        return 0.0

    text_lower = text.lower().strip()
    target_lower = target_title.lower().strip()

    if target_lower in text_lower:
        return 1.0

    target_words = set(target_lower.split())
    text_words = set(text_lower.split())
    if target_words and target_words.issubset(text_words):
        return 1.0

    word_ratio = 0.0
    if target_words:
        matched = sum(1 for w in target_words if any(w in tw for tw in text_words))
        word_ratio = matched / len(target_words)

    full_ratio = SequenceMatcher(None, text_lower, target_lower).ratio()

    window_size = len(target_lower)
    best_window = 0.0
    if len(text_lower) >= window_size:
        for i in range(len(text_lower) - window_size + 1):
            chunk = text_lower[i:i + window_size]
            r = SequenceMatcher(None, chunk, target_lower).ratio()
            if r > best_window:
                best_window = r

    return max(full_ratio, word_ratio, best_window)


def ocr_spine_crop(img, corners, target_title, reader=None):
    """
    Warps the OBB region upright, OCRs it (trying both rotation
    directions for vertical spines, since the correct reading direction
    varies), and returns (best_text, similarity_score).
    """
    reader = reader or get_reader()
    crop = warp_obb_to_upright(img, corners)

    if crop.shape[0] > crop.shape[1]:  # taller than wide = vertical spine
        crop_cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        crop_ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        text_cw = " ".join(reader.readtext(crop_cw, detail=0)).strip()
        text_ccw = " ".join(reader.readtext(crop_ccw, detail=0)).strip()

        ratio_cw = title_similarity(text_cw, target_title) if text_cw else 0.0
        ratio_ccw = title_similarity(text_ccw, target_title) if text_ccw else 0.0

        text = text_cw if ratio_cw >= ratio_ccw else text_ccw
    else:
        text = " ".join(reader.readtext(crop, detail=0)).strip()

    similarity = title_similarity(text, target_title) if text else 0.0
    return text, similarity
