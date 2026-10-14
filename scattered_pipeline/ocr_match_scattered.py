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


def get_reader(languages=("ch_sim", "en"), verbose=True):
    """Lazy-load EasyOCR once via shared singleton, reused across calls."""
    try:
        from shared.ocr_reader import get_shared_ocr_reader
        return get_shared_ocr_reader(languages=languages, verbose=verbose)
    except Exception:
        global _ocr_reader
        if _ocr_reader is None:
            if verbose:
                print(f"[Scattered OCR] Loading EasyOCR {languages}...", flush=True)
            _ocr_reader = easyocr.Reader(list(languages), gpu=(DEVICE == "cuda"))
            if verbose:
                print("[Scattered OCR] EasyOCR ready.", flush=True)
        return _ocr_reader


def preload_scattered_reader(languages=("ch_sim", "en"), verbose=False, **kwargs):
    """Force-load the scattered OCR reader immediately — call from a
    background thread at startup so it's ready by the time OCR needs it."""
    get_reader(languages, verbose=verbose)


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


from shared.ocr_reader import compute_title_similarity

# Alias for compatibility with spine pipeline
title_similarity = compute_title_similarity


def ocr_best_match(crop, query_title, reader=None, early_exit_score=0.80):
    """
    Fast orientation-aware OCR matching:
    1. Resizes crop to max_dim 500 for optimal PP-OCR speed (~3x faster than 900px).
    2. Tests primary orientations (0° and 90° CW) leveraging RapidOCR's internal 180° cls.
    3. Exits immediately once a strong match (>= 0.80) is found without testing remaining rotations.
    """
    if crop is None or crop.size == 0:
        return 0.0, ""

    reader = reader or get_reader()

    max_dim = max(crop.shape[:2])
    if max_dim > 500:
        scale = 500.0 / max_dim
        crop_resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    else:
        crop_resized = crop

    # Primary orientations (horizontal vs vertical)
    h, w = crop_resized.shape[:2]
    if w >= h:
        primary_rots = [crop_resized, cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE)]
        secondary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_180), cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    else:
        primary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE), crop_resized]
        secondary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(crop_resized, cv2.ROTATE_180)]

    best_score = 0.0
    best_text = ""

    # Phase 1: Test primary orientations
    for rot_crop in primary_rots:
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
                return best_score, best_text

    # Phase 2: Fallback to secondary orientations only if no candidate reached 0.60
    if best_score < 0.60:
        for rot_crop in secondary_rots:
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
