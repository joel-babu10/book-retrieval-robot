"""
shared/mode_classifier.py
----------------------------------------------------------------------------
Ultra-lightweight Scene Arrangement Classifier (3.0 MB).
Automatically classifies camera RGB frame into "spine" vs "scattered"
in ~2ms so the pipeline loads/uses ONLY the needed specialized OBB model.
"""

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

    # 1. Detection checks on camera frame
    img_check = cv_image

    # 1a. Detect shelf spines first
    tall_spines = []
    try:
        from spine_pipeline.detect_spine import detect_spines
        _, spine_dets = detect_spines(image=img_check, conf_threshold=0.25)
        for d in spine_dets:
            c = d["corners"]
            w_ext = float(np.ptp(c[:, 0]))
            h_ext = float(np.ptp(c[:, 1]))
            # Shelf spines are vertically oriented (height > 1.10 * width)
            if h_ext > 1.10 * w_ext and h_ext >= 50.0:
                tall_spines.append(d)
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Spine check error: {e}", flush=True)

    # 1b. Detect books and categorize upright vs flat
    upright_books = []
    flat_books = []
    try:
        from scattered_pipeline.detect_scattered import detect_books
        _, sc_dets = detect_books(image=img_check, conf_threshold=0.35)
        for d in sc_dets:
            w_box = float(min(d['width_px'], d['height_px']))
            h_box = float(max(d['width_px'], d['height_px']))
            aspect = h_box / max(w_box, 1.0)
            ang = abs(float(d.get('obb_angle', 0.0)))
            is_upright = (aspect >= 1.5) or (ang <= 30.0 or ang >= 150.0 or abs(ang - 90.0) <= 20.0)

            # Check if this book candidate overlaps an already identified shelf spine
            sp_cx, sp_cy = d["center"]
            overlap = False
            for sp in tall_spines:
                scx, scy = sp["center"]
                if np.hypot(sp_cx - scx, sp_cy - scy) < 150.0:
                    overlap = True
                    break
                fc = d.get("obb_corners")
                if fc is not None and cv2.pointPolygonTest(np.array(fc, dtype=np.float32), (float(scx), float(scy)), False) >= 0:
                    overlap = True
                    break

            if overlap or is_upright:
                upright_books.append(d)
            else:
                flat_books.append(d)
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Scattered check error: {e}", flush=True)

    # 2. Arrangement determination
    n_spines = len(tall_spines)
    n_upright = len(upright_books)
    n_scattered = len(flat_books)

    # When vertical shelf spines or upright standing books are present, scene is SPINE arrangement
    if n_spines >= 1 or n_upright >= 1:
        total_standing = max(n_spines, n_upright)
        if n_scattered >= 2 and n_scattered > total_standing:
            mode = "mixed"
            conf = 0.90
            if verbose:
                print(f"[Mode Classifier] Detected {total_standing} upright book/spine(s) AND {n_scattered} flat surface book(s). Scene is MIXED -> Mode set to: 'MIXED'", flush=True)
            return mode, conf
        else:
            mode = "spine"
            conf = 0.95
            if verbose:
                print(f"[Mode Classifier] Detected {total_standing} upright shelf book/spine(s). Mode set to: 'SPINE'", flush=True)
            return mode, conf

    # Desk-only scattered scene (no spines or upright books, only flat books)
    if n_scattered >= 1:
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
        except Exception:
            pass

    mode, conf = _heuristic_mode_fallback(cv_image)
    if verbose:
        print(f"[Mode Classifier] Heuristic classification: '{mode.upper()}' (conf: {conf:.2f})", flush=True)
    return mode, conf
