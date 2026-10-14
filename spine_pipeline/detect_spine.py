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
CONF_THRESHOLD = 0.25

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


def preload_spine_model(model_path=None, verbose=False):
    """Force-load the spine YOLO model immediately."""
    get_model(model_path=model_path)


def detect_spines(image_path=None, image=None, conf_threshold=CONF_THRESHOLD, model_path=None, refine_sam=False):
    """
    Runs YOLO OBB spine detection followed by SAM mask edge refinement.

    Workflow:
      Stage 1: YOLO OBB Detection -> generates candidate oriented bounding boxes.
      Stage 2: SAM Mask Refinement -> refines bounding box contour edges.
      Stage 3: Feature Extraction  -> computes centroids, spine width, and top-edge points.

    Returns:
        img: the loaded/used image (np.ndarray, BGR)
        detections: list of dicts, each with:
            corners         - 4x2 np.array of OBB corner points (SAM-refined if refine_sam=True)
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

    if results.obb is None or len(results.obb) == 0:
        return img, []

    # 1. Gather raw YOLO OBB candidates
    raw_candidates = []
    for obb in results.obb:
        conf = obb.conf[0].item()
        if conf < conf_threshold:
            continue
        corners = obb.xyxyxyxy[0].cpu().numpy()
        xywhr = obb.xywhr[0].cpu().numpy()
        raw_candidates.append({
            "corners": corners,
            "xywhr": xywhr,
            "conf": conf,
        })

    if not raw_candidates:
        return img, []

    # 2. Refine OBB corners using SAM if enabled (batch ViT encoding for high efficiency)
    if refine_sam:
        try:
            from shared.sam_refine import refine_obbs_with_sam
            corners_list = [item["corners"] for item in raw_candidates]
            refined_corners_list = refine_obbs_with_sam(img, corners_list)
            for item, ref_corners in zip(raw_candidates, refined_corners_list):
                item["corners"] = ref_corners
        except Exception as e:
            print(f"[Spine] SAM refinement fallback to YOLO OBB: {e}", flush=True)

    # 3. Compute final box centers, spine widths, and top edge centroids from refined corners
    detections = []
    for item in raw_candidates:
        corners = item["corners"]
        conf = item["conf"]
        xywhr = item["xywhr"]

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
