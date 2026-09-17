"""
spine_pipeline/detect_spine.py
----------------------------------------------------------------------------
YOLO OBB spine detection — detection only, no OCR. Kept separate so
detection can be tested/tuned independently of text matching.
"""

from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_MODEL_PATH = r"C:\Users\joelv\runs\obb\train-3\weights\best.pt"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent.parent / "yolo26s-obb.pt"
CONF_THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def get_model(model_path=None):
    """Lazy-load the YOLO model once, reused across calls."""
    global _model
    if _model is None:
        if model_path is None:
            if Path(DEFAULT_MODEL_PATH).exists():
                model_path = DEFAULT_MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                model_path = str(FALLBACK_MODEL_PATH)
            else:
                model_path = DEFAULT_MODEL_PATH

        print(f"[Spine] Loading YOLO OBB model ({DEVICE}): {model_path}", flush=True)
        _model = YOLO(model_path)
    return _model


def detect_spines(image_path=None, image=None, conf_threshold=CONF_THRESHOLD, model_path=None):
    """
    Runs OBB spine detection on either image_path (file) or image (numpy
    array, e.g. live camera frame).

    Returns:
        img: the loaded/used image (np.ndarray, BGR)
        detections: list of dicts, each with:
            corners         - 4x2 np.array of OBB corner points
            xywhr           - raw [x, y, w, h, r] from YOLO OBB
            conf            - detection confidence
            center          - (u, v) box centroid
            width_px        - spine width in pixels (short side of the box)
            top_edge_center - (u, v) midpoint of the box's top edge
    """
    model = get_model(model_path)

    if image is not None:
        img = image
        results = model(img, imgsz=640, device=DEVICE, verbose=False)[0]
    else:
        results = model(image_path, imgsz=640, device=DEVICE, verbose=False)[0]
        import cv2
        img = cv2.imread(image_path)

    detections = []

    if results.obb is None:
        return img, detections

    for obb in results.obb:
        conf = obb.conf[0].item()
        if conf < conf_threshold:
            continue

        corners = obb.xyxyxyxy[0].cpu().numpy()
        xywhr = obb.xywhr[0].cpu().numpy()

        center_u = float(np.mean(corners[:, 0]))
        center_v = float(np.mean(corners[:, 1]))
        spine_width_px = float(min(xywhr[2], xywhr[3]))

        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_two = sorted_by_y[:2]
        top_edge_center_u = float(np.mean(top_two[:, 0]))
        top_edge_center_v = float(np.mean(top_two[:, 1]))

        detections.append({
            "corners": corners,
            "xywhr": xywhr,
            "conf": conf,
            "center": (center_u, center_v),
            "width_px": spine_width_px,
            "top_edge_center": (top_edge_center_u, top_edge_center_v),
        })

    return img, detections
