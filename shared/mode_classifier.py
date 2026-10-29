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

    upright_books = []
    flat_books = []

    # 1. Primary detection: Detect books using Book YOLO model
    try:
        from scattered_pipeline.detect_scattered import detect_books
        _, sc_dets = detect_books(image=cv_image, conf_threshold=0.35)
        for d in sc_dets:
            c = d["corners"]
            dx = float(np.ptp(c[:, 0]))
            dy = float(np.ptp(c[:, 1]))
            w_box = float(min(d['width_px'], d['height_px']))
            h_box = float(max(d['width_px'], d['height_px']))
            aspect = h_box / max(w_box, 1.0)
            cx, cy = d["center"]
            y_top = float(np.min(c[:, 1]))

            # Upright shelf book criteria:
            # - Must be tall in image (dy >= 350 px in 1080p)
            # - Must reach up into the shelf elevation (y_top < 550 px)
            # - Must have vertical span much larger than horizontal span (dy >= 1.8 * dx)
            # - Must have high aspect ratio (aspect >= 2.0)
            is_tall_shelf = (dy >= 350.0 and y_top < 550.0 and dy >= 1.8 * dx and aspect >= 2.0)

            # Flat surface book criteria:
            # - Horizontal span wider than vertical span (dx >= 1.15 * dy)
            # - Wide cover face visible (w_box >= 170 px and aspect <= 2.2)
            # - Desk surface location (cy > 750 or y_top > 550) with dy < 350
            # - Any book not meeting strict upright shelf criteria
            is_flat_desk = (dx >= 1.15 * dy) or (w_box >= 170.0 and aspect <= 2.2) or (cy > 750.0 and dy < 350.0) or (y_top > 550.0 and not is_tall_shelf)

            if is_tall_shelf and not is_flat_desk:
                upright_books.append(d)
            else:
                flat_books.append(d)
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Book detection check error: {e}", flush=True)

    # 2. Check for shelf spines (only count true vertical shelf spines, not desk floor slices)
    tall_spines = []
    try:
        from spine_pipeline.detect_spine import detect_spines
        _, spine_dets = detect_spines(image=cv_image, conf_threshold=0.35)
        for d in spine_dets:
            c = d["corners"]
            dx = float(np.ptp(c[:, 0]))
            dy = float(np.ptp(c[:, 1]))
            cx, cy = d["center"]
            y_top = float(np.min(c[:, 1]))

            # A real shelf spine is a tall vertical column reaching shelf height
            if dy >= 350.0 and dy >= 2.5 * dx and y_top < 550.0 and cy < 820.0:
                # Must not overlap or be inside any detected flat book
                overlap_flat = False
                for fb in flat_books:
                    fc = fb["corners"]
                    fcx, fcy = fb["center"]
                    if np.hypot(cx - fcx, cy - fcy) < 160.0:
                        overlap_flat = True
                        break
                    if cv2.pointPolygonTest(np.array(fc, dtype=np.float32), (float(cx), float(cy)), False) >= 0:
                        overlap_flat = True
                        break

                # Must not duplicate an already counted upright book
                overlap_upright = False
                for ub in upright_books:
                    ucx, ucy = ub["center"]
                    if np.hypot(cx - ucx, cy - ucy) < 140.0:
                        overlap_upright = True
                        break

                if not overlap_flat and not overlap_upright:
                    tall_spines.append(d)
    except Exception as e:
        if verbose:
            print(f"[Mode Classifier] Spine check error: {e}", flush=True)

    # 3. Scene arrangement determination
    n_upright = len(upright_books) + len(tall_spines)
    n_flat = len(flat_books)

    # When both upright shelf books/spines AND flat surface books are present -> MIXED arrangement
    if n_upright >= 1 and n_flat >= 1:
        mode = "mixed"
        conf = 0.95
        if verbose:
            print(f"[Mode Classifier] Detected {n_upright} upright shelf book/spine(s) AND {n_flat} flat surface book(s). Mode set to: 'MIXED'", flush=True)
        return mode, conf

    # Only flat surface books detected on desk -> SCATTERED arrangement
    if n_flat >= 1 and n_upright == 0:
        mode = "scattered"
        conf = 0.95
        if verbose:
            print(f"[Mode Classifier] Detected {n_flat} flat surface book(s) and 0 upright shelf spines. Mode set to: 'SCATTERED'", flush=True)
        return mode, conf

    # Only upright shelf spines detected -> SPINE arrangement
    if n_upright >= 1 and n_flat == 0:
        mode = "spine"
        conf = 0.95
        if verbose:
            print(f"[Mode Classifier] Detected {n_upright} upright shelf spine(s) and 0 flat surface books. Mode set to: 'SPINE'", flush=True)
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
