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


def get_reader(languages=OCR_LANGUAGES, gpu=USE_GPU, verbose=True):
    """Lazy-load EasyOCR once via shared singleton, reused across calls."""
    try:
        from shared.ocr_reader import get_shared_ocr_reader
        return get_shared_ocr_reader(languages=tuple(languages), gpu=gpu, verbose=verbose)
    except Exception:
        global _reader
        if _reader is None:
            if verbose:
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

    # Inset left and right edges slightly (5%) to avoid text bleed from adjacent books on the shelf
    inset_t = (tr - tl) * 0.05
    inset_b = (br - bl) * 0.05
    tl_in = tl + inset_t
    tr_in = tr - inset_t
    bl_in = bl + inset_b
    br_in = br - inset_b

    src_pts = np.array([tl_in, tr_in, br_in, bl_in], dtype=np.float32)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (out_w, out_h))


from shared.ocr_reader import compute_title_similarity

# Alias for compatibility
title_similarity = compute_title_similarity


def ocr_spine_crop(img, corners, target_title, reader=None, early_exit_score=0.80):
    """
    Fast orientation-aware spine OCR matching:
    1. Warps OBB spine region upright and scales to max_dim 450.
    2. Tests primary orientations (horizontal and vertical) first.
    3. Exits immediately once a match (>= 0.80) is found.
    """
    reader = reader or get_reader()
    crop = warp_obb_to_upright(img, corners)
    if crop is None or crop.size == 0:
        return "", 0.0

    max_dim = max(crop.shape[:2])
    if max_dim > 450:
        scale = 450.0 / max_dim
        crop_resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    else:
        crop_resized = crop

    h, w = crop_resized.shape[:2]
    if h > w:
        primary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE), crop_resized]
        secondary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(crop_resized, cv2.ROTATE_180)]
    else:
        primary_rots = [crop_resized, cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE)]
        secondary_rots = [cv2.rotate(crop_resized, cv2.ROTATE_180), cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)]

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

        score = title_similarity(read_text, target_title)
        if score > best_score:
            best_score = score
            best_text = read_text
            if best_score >= early_exit_score:
                return best_text, best_score

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

            score = title_similarity(read_text, target_title)
            if score > best_score:
                best_score = score
                best_text = read_text
                if best_score >= early_exit_score:
                    break

    return best_text, best_score

