"""
scattered_pipeline/ocr_match_scattered.py
----------------------------------------------------------------------------
OCR + title-matching logic for detected scattered books. Kept separate
from detection so OCR tuning doesn't touch the SAM 3 detection code.
"""

import sys
import cv2
import numpy as np
import torch
import easyocr
from difflib import SequenceMatcher

# Force UTF-8 output encoding for Chinese character printing on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ocr_reader = None


def get_reader(languages=("ch_sim", "en")):
    """Lazy-load EasyOCR once, reused across calls."""
    global _ocr_reader
    if _ocr_reader is None:
        print(f"[Scattered OCR] Loading EasyOCR {languages}...", flush=True)
        _ocr_reader = easyocr.Reader(list(languages), gpu=(DEVICE == "cuda"))
        print("[Scattered OCR] EasyOCR ready.", flush=True)
    return _ocr_reader


def preload_scattered_reader(languages=("ch_sim", "en")):
    """Force-load the scattered OCR reader immediately — call from a
    background thread at startup so it's ready by the time OCR needs it."""
    get_reader(languages)


def warp_obb_to_upright(img, corners):
    """Warps an oriented bounding box (OBB) directly into an upright rectangle."""
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


def compute_title_similarity(ocr_text, query_title):
    """
    Robust title similarity matching supporting English & Chinese:
    1. Exact substring match
    2. Word token set containment
    3. Word overlap ratio
    4. Sliding window substring ratio (prevents long cover text from diluting title score)
    5. SequenceMatcher full ratio
    """
    if not ocr_text or not query_title:
        return 0.0

    text_lower = ocr_text.lower().strip()
    query_lower = query_title.lower().strip()

    if query_lower in text_lower:
        return 1.0

    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    if query_words and query_words.issubset(text_words):
        return 1.0

    word_ratio = 0.0
    if query_words:
        matched = sum(1 for w in query_words if any(w in tw for tw in text_words))
        word_ratio = matched / len(query_words)

    full_ratio = SequenceMatcher(None, text_lower, query_lower).ratio()

    window_size = len(query_lower)
    best_window = 0.0
    if len(text_lower) >= window_size:
        for i in range(len(text_lower) - window_size + 1):
            chunk = text_lower[i:i + window_size]
            r = SequenceMatcher(None, chunk, query_lower).ratio()
            if r > best_window:
                best_window = r

    return max(full_ratio, word_ratio, best_window)


# Alias for compatibility with spine pipeline
title_similarity = compute_title_similarity


def ocr_best_match(crop, query_title, reader=None, early_exit_score=0.85):
    """
    Tries OCR at rotations (cover orientation is unconstrained),
    resizing large crops first for speed, ordering rotation attempts
    smartly based on aspect ratio, and returning the best (score, text) pair.
    Exits early once a strong match is found.
    """
    if crop is None or crop.size == 0:
        return 0.0, ""

    reader = reader or get_reader()

    max_dim = max(crop.shape[:2])
    if max_dim > 640:
        scale = 640.0 / max_dim
        crop_resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    else:
        crop_resized = crop

    # Order rotations intelligently based on aspect ratio
    h, w = crop_resized.shape[:2]
    if w >= h:
        # Landscape crop: check 0 and 180 degrees first
        rotations = [
            crop_resized,
            cv2.rotate(crop_resized, cv2.ROTATE_180),
            cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
    else:
        # Portrait crop: check 0, 90 CW, 90 CCW, 180 degrees
        rotations = [
            crop_resized,
            cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(crop_resized, cv2.ROTATE_180),
        ]

    best_score = 0.0
    best_text = ""

    for rot_crop in rotations:
        try:
            results = reader.readtext(rot_crop, detail=0)
        except Exception:
            continue
        read_text = " ".join(results).strip()
        if not read_text:
            continue

        score = compute_title_similarity(read_text, query_title)
        if score > best_score:
            best_score = score
            best_text = read_text
            if best_score >= early_exit_score:
                break

    return best_score, best_text


def ocr_scattered_crop(img, corners, target_title, reader=None, early_exit_score=0.85):
    """
    Convenience function: warps OBB region upright and runs multi-rotation OCR matching.
    Returns (text, similarity_score).
    """
    crop = warp_obb_to_upright(img, corners)
    score, text = ocr_best_match(crop, target_title, reader=reader, early_exit_score=early_exit_score)
    return text, score
