"""
shared/mode_classifier.py
----------------------------------------------------------------------------
Ultra-lightweight Scene Arrangement Classifier (3.0 MB).
Automatically classifies camera RGB frame into "spine" vs "scattered"
in ~2ms so the pipeline loads/uses ONLY the needed specialized OBB model.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "mode_classifier.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_classifier_model = None


def get_mode_classifier(verbose=True):
    """Lazy-load the lightweight mode classifier once (reused across calls)."""
    global _classifier_model
    if _classifier_model is not None:
        return _classifier_model

    if not MODEL_PATH.exists():
        if verbose:
            print(f"[Mode Classifier] Notice: {MODEL_PATH} not found. Will use heuristic fallback.", flush=True)
        return None

    try:
        from ultralytics import YOLO
        if verbose:
            print(f"[Mode Classifier] Loading tiny arrangement classifier ({MODEL_PATH.name})...", flush=True)
        _classifier_model = YOLO(str(MODEL_PATH))
        if verbose:
            print("[Mode Classifier] Classifier ready.", flush=True)
        return _classifier_model
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Warning: Failed to load classifier: {e}", flush=True)
        return None


def preload_mode_classifier(verbose=False):
    """Pre-warm classifier in background thread."""
    get_mode_classifier(verbose=verbose)


def _heuristic_mode_fallback(cv_image):
    """
    Fallback edge-density heuristic if model weights are unavailable:
    Spines have dense, parallel vertical edges along the vertical axis.
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_x = np.mean(np.abs(sobel_x))
    mag_y = np.mean(np.abs(sobel_y))
    # Vertical edge dominance -> spine
    ratio = mag_x / max(mag_y, 1e-5)
    if ratio > 1.25:
        return "spine", 0.70
    return "scattered", 0.70


def classify_mode(cv_image, verbose=True):
    """
    Classifies the scene arrangement from a single color image:
      - 'spine'     : Only vertical shelf spines detected
      - 'scattered' : Only flat surface books detected
      - 'mixed'     : BOTH vertical spines and flat books detected in the same scene!

    Returns:
        mode: "spine", "scattered", or "mixed"
        confidence: float between 0.0 and 1.0
    """
    if cv_image is None or cv_image.size == 0:
        return "spine", 0.50

    tall_spines = []
    flat_books = []

    # 1. Fast detection checks on 640px resized frame (~15 ms total on GPU)
    h, w = cv_image.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        img_check = cv2.resize(cv_image, (int(w * scale), int(h * scale)))
    else:
        img_check = cv_image

    try:
        from scattered_pipeline.detect_scattered import detect_books
        _, sc_dets = detect_books(image=img_check, conf_threshold=0.40)
        flat_books = [
            d for d in sc_dets
            if ((d['height_px'] / max(d['width_px'], 1.0)) < 2.5)
        ]
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Scattered check error: {e}", flush=True)

    try:
        from spine_pipeline.detect_spine import detect_spines
        _, spine_dets = detect_spines(image=img_check, conf_threshold=0.45)
        tall_spines = []
        for d in spine_dets:
            c = d["corners"]
            w_ext = float(np.ptp(c[:, 0]))
            h_ext = float(np.ptp(c[:, 1]))
            # A true shelf spine MUST stand vertically (height > 1.5 * width)
            # Horizontal page edges of flat books have width > height and are NOT shelf spines
            if h_ext > 1.5 * w_ext and h_ext >= 60.0:
                # Also ensure this spine candidate does NOT sit inside a detected flat book
                sp_cx, sp_cy = d["center"]
                inside_flat = False
                for fb in flat_books:
                    fc = fb.get("obb_corners")
                    if fc is not None:
                        if cv2.pointPolygonTest(np.array(fc, dtype=np.float32), (float(sp_cx), float(sp_cy)), False) >= 0:
                            inside_flat = True
                            break
                if not inside_flat:
                    tall_spines.append(d)
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Spine check error: {e}", flush=True)

    # 2. Arrangement determination
    n_spines = len(tall_spines)
    n_scattered = len(flat_books)

    # Mixed scene: both vertical shelf spines and flat surface books present
    if n_spines >= 1 and n_scattered >= 1:
        mode = "mixed"
        conf = 0.98
        if verbose:
            print(f"[Mode Classifier] Detected {n_spines} upright shelf spine(s) AND {n_scattered} flat book(s). Scene is MIXED -> Mode set to: 'MIXED'", flush=True)
        return mode, conf

    # Shelf-only scene
    if n_spines >= 1 and n_scattered == 0:
        mode = "spine"
        conf = 0.95
        if verbose:
            print(f"[Mode Classifier] Detected {n_spines} vertical shelf spine(s) and 0 flat books. Mode set to: 'SPINE'", flush=True)
        return mode, conf

    # Desk-only scattered scene
    if n_spines == 0 and n_scattered >= 1:
        mode = "scattered"
        conf = 0.95
        if verbose:
            print(f"[Mode Classifier] Detected 0 vertical spines and {n_scattered} flat book(s). Mode set to: 'SCATTERED'", flush=True)
        return mode, conf

    # Fallback to tiny classifier if geometry is inconclusive
    model = get_mode_classifier(verbose=verbose)
    if model is not None:
        try:
            results = model(cv_image, imgsz=224, verbose=False, device=DEVICE)[0]
            top_class = results.names[results.probs.top1]
            conf = float(results.probs.top1conf)
            if verbose:
                print(f"[Mode Classifier] Geometry inconclusive. Tiny classifier predicts: '{top_class.upper()}' (conf: {conf:.2f})", flush=True)
            return top_class, conf
        except Exception as e:
            pass

    mode, conf = _heuristic_mode_fallback(cv_image)
    if verbose:
        print(f"[Mode Classifier] Heuristic classification: '{mode.upper()}' (conf: {conf:.2f})", flush=True)
    return mode, conf
