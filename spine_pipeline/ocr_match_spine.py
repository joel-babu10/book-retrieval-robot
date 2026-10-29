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


def warp_obb_to_upright(img, corners, expand_width_px=0):
    """
    Takes 4 corner points of a rotated box and returns a straightened
    (de-rotated) crop of just that spine, upright and rectangular.
    Supports optional width expansion to prevent slicing through text on spine creases.
    """
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_pts = sorted_by_y[:2]
    bottom_pts = sorted_by_y[2:]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    tl, tr = top_pts
    bl, br = bottom_pts

    w_vec = tr - tl
    w_len = np.linalg.norm(w_vec)
    unit_w = w_vec / max(w_len, 1e-4)

    # Pad width: expand outward to avoid cutting off letters at spine boundaries/creases
    pad = float(expand_width_px)
    tl_out = tl - unit_w * (pad * 0.35)
    tr_out = tr + unit_w * (pad * 0.85)
    bl_out = bl - unit_w * (pad * 0.35)
    br_out = br + unit_w * (pad * 0.85)

    out_w = max(int(np.linalg.norm(tr_out - tl_out)), 1)
    out_h = max(int(np.linalg.norm(bl_out - tl_out)), 1)

    src_pts = np.array([tl_out, tr_out, br_out, bl_out], dtype=np.float32)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (out_w, out_h))


from shared.ocr_reader import compute_title_similarity

# Alias for compatibility
title_similarity = compute_title_similarity


def ocr_spine_crop(img, corners, target_title, reader=None, early_exit_score=0.80):
    """
    Fast orientation-aware spine OCR matching:
    1. Warps OBB spine region upright (testing nominal and expanded width crops).
    2. Tests primary orientations (CCW, CW, upright, 180).
    3. Exits immediately once a confident match (>= 0.80) is found.
    """
    reader = reader or get_reader()

    # Determine frame scale: expand crop more on 1080p than on 480p
    img_w = img.shape[1] if img is not None else 640
    nominal_pad = max(8, int(round(12 * (img_w / 640.0))))
    wide_pad = max(25, int(round(55 * (img_w / 640.0))))

    best_score = 0.0
    best_text = ""
    most_descriptive_text = ""

    # Phase 1 & 2: Test nominal crop, then expanded-width crop if confidence is low
    pads_to_try = [nominal_pad, wide_pad]

    for pad_x in pads_to_try:
        crop = warp_obb_to_upright(img, corners, expand_width_px=pad_x)
        if crop is None or crop.size == 0:
            continue

        max_dim = max(crop.shape[:2])
        if max_dim > 1080:
            scale = 1080.0 / max_dim
            crop_resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        else:
            crop_resized = crop

        h, w = crop_resized.shape[:2]
        # For vertical shelf spines (h > w), text is rotated either 90 deg CCW or CW, or runs upright
        if h > w:
            rotations = [
                cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE),
                crop_resized,
                cv2.rotate(crop_resized, cv2.ROTATE_180),
            ]
        else:
            rotations = [
                crop_resized,
                cv2.rotate(crop_resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(crop_resized, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(crop_resized, cv2.ROTATE_180),
            ]

        for rot_crop in rotations:
            try:
                results = reader.readtext(rot_crop, detail=0)
            except Exception:
                continue
            read_text = " ".join(results).strip()
            if not read_text:
                continue

            if len(read_text) > len(most_descriptive_text):
                most_descriptive_text = read_text

            score = title_similarity(read_text, target_title)
            if score > best_score:
                best_score = score
                best_text = read_text
                if best_score >= early_exit_score:
                    return best_text, best_score

        # If already matched strongly (>= 0.60), no need to try wide padding
        if best_score >= 0.60:
            break

    # If not matched confidently yet, try CLAHE contrast enhancement for low-contrast/aged spine text
    if best_score < early_exit_score:
        try:
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            clahe_rotations = [
                cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE) if h > w else enhanced,
                cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE) if h > w else cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]
            for rot_crop in clahe_rotations:
                try:
                    results = reader.readtext(rot_crop, detail=0)
                except Exception:
                    continue
                read_text = " ".join(results).strip()
                if not read_text:
                    continue
                if len(read_text) > len(most_descriptive_text):
                    most_descriptive_text = read_text
                score = title_similarity(read_text, target_title)
                if score > best_score:
                    best_score = score
                    best_text = read_text
                    if best_score >= early_exit_score:
                        return best_text, best_score
        except Exception:
            pass

    # Return best matching text if any similarity > 0, otherwise return the text actually read on the book
    final_text = best_text if (best_text and best_score > 0.0) else most_descriptive_text
    return final_text, best_score

