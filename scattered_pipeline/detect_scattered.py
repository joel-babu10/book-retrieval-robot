"""
scattered_pipeline/detect_scattered.py
----------------------------------------------------------------------------
SAM 3 text-prompted book segmentation + book-only filtering. Kept separate
from OCR matching so detection can be tuned independently.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure local sam3 repository is in sys.path if present
SAM3_ROOT = Path(__file__).resolve().parent.parent / "sam3"
if SAM3_ROOT.exists() and str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

TEXT_PROMPT = "book"
CONF_THRESHOLD = 0.35

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_sam3 = None
_processor_sam3 = None


YOLO_SCATTERED_PATH = r"C:\Users\joelv\runs\obb\runs\obb\scattered-train-2-3\weights\best.pt"
SAM1_CHECKPOINT = r"sam_vit_b_01ec64.pth"

_yolo_scattered_model = None
_sam_predictor_fallback = None


def get_yolo_scattered(verbose=True):
    """Loads trained YOLO OBB book model (5.9 MB, ~35 ms inference)."""
    global _yolo_scattered_model
    if _yolo_scattered_model is not None:
        return _yolo_scattered_model

    if os.path.exists(YOLO_SCATTERED_PATH):
        from ultralytics import YOLO
        if verbose:
            print(f"[Scattered] Loading trained Book YOLO OBB model ({Path(YOLO_SCATTERED_PATH).name})...", flush=True)
        _yolo_scattered_model = YOLO(YOLO_SCATTERED_PATH)
        if verbose:
            print("[Scattered] Book YOLO OBB model ready.", flush=True)
        return _yolo_scattered_model
    return None


def get_sam3(conf_threshold=CONF_THRESHOLD, verbose=True):
    """Lazy-load SAM 3 once, reused across calls. Prefers a local cached
    checkpoint over re-downloading from Hugging Face."""
    global _model_sam3, _processor_sam3
    if _model_sam3 is not None:
        return _model_sam3, _processor_sam3

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if verbose:
        print(f"[Scattered] Device: {DEVICE}", flush=True)
        print("[Scattered] Loading SAM 3...", flush=True)

    local_hf_ckpt = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
        "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
    )
    custom_ckpt = os.environ.get("SAM3_CHECKPOINT", None)
    ckpt_to_use = None
    if custom_ckpt and os.path.exists(custom_ckpt):
        ckpt_to_use = custom_ckpt
    elif os.path.exists(local_hf_ckpt):
        ckpt_to_use = local_hf_ckpt

    if ckpt_to_use:
        if verbose:
            print(f"[Scattered] Loading SAM 3 locally from: {ckpt_to_use}", flush=True)
        _model_sam3 = build_sam3_image_model(device=DEVICE, checkpoint_path=ckpt_to_use, load_from_HF=False)
    else:
        if verbose:
            print("[Scattered] Loading SAM 3 from Hugging Face...", flush=True)
        _model_sam3 = build_sam3_image_model(device=DEVICE)

    _processor_sam3 = Sam3Processor(_model_sam3, device=DEVICE, confidence_threshold=conf_threshold)
    if verbose:
        print("[Scattered] SAM 3 ready.", flush=True)
    return _model_sam3, _processor_sam3


def preload_scattered_model(conf_threshold=CONF_THRESHOLD, verbose=True, **kwargs):
    """Preload scattered book detector in background thread."""
    if os.environ.get("USE_SAM3", "0") == "1":
        try:
            get_sam3(conf_threshold=conf_threshold, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[Scattered] SAM 3 preload warning: {e}", flush=True)
            get_yolo_scattered(verbose=verbose)
    else:
        get_yolo_scattered(verbose=verbose)


def _bbox_containment_ratio(rect_child, rect_parent):
    """Quick 2D bounding box intersection-over-child-area check."""
    xc, yc, wc, hc = rect_child
    xp, yp, wp, hp = rect_parent
    x_overlap = max(0, min(xc + wc, xp + wp) - max(xc, xp))
    y_overlap = max(0, min(yc + hc, yp + hp) - max(yc, yp))
    overlap_area = x_overlap * y_overlap
    child_area = wc * hc
    if child_area == 0:
        return 0.0
    return overlap_area / float(child_area)


def filter_book_detections(detections, img_width, img_height,
                            min_area_ratio=0.003,
                            max_area_ratio=0.85,
                            max_aspect_ratio=6.0,
                            containment_thresh=0.85):
    """
    Strict book-only filter:
    1. Removes full-screen background frames (> 85% image area).
    2. Removes tiny noise fragments (< 0.3% image area).
    3. Removes extreme line aspect ratios (> 6.0).
    4. Suppresses nested child boxes (text/logo/graphics inside a cover)
       without dropping touching/adjacent books. Optimized with fast BBox pre-filtering.
    """
    img_area = float(img_width * img_height)
    candidates = []

    for d in detections:
        (cx, cy), (w, h), angle = d["obb_center"], d["obb_size"], d["obb_angle"]
        box_area = w * h
        area_ratio = box_area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        short_side, long_side = min(w, h), max(w, h)
        if short_side <= 0:
            continue
        if long_side / short_side > max_aspect_ratio:
            continue

        # Cache bounding rect for fast containment checks
        d["bounding_rect"] = cv2.boundingRect(d["mask_contour"])
        candidates.append(d)

    candidates.sort(key=lambda d: d["obb_size"][0] * d["obb_size"][1], reverse=True)

    filtered = []
    for d in candidates:
        contour_d = d["mask_contour"]
        rect_d = d["bounding_rect"]
        is_nested = False

        for parent in filtered:
            rect_p = parent["bounding_rect"]
            # Fast-path check: if bounding box overlap ratio is low, contour can't be nested
            bbox_ratio = _bbox_containment_ratio(rect_d, rect_p)
            if bbox_ratio < (containment_thresh * 0.7):
                continue

            # Crop region of interest around parent bounding box for fast contour mask check
            xp, yp, wp, hp = rect_p
            if wp <= 0 or hp <= 0:
                continue

            mask_d = np.zeros((hp, wp), dtype=np.uint8)
            mask_p = np.zeros((hp, wp), dtype=np.uint8)

            contour_d_offset = contour_d - np.array([xp, yp])
            contour_p_offset = parent["mask_contour"] - np.array([xp, yp])

            cv2.drawContours(mask_d, [contour_d_offset], -1, 1, -1)
            cv2.drawContours(mask_p, [contour_p_offset], -1, 1, -1)

            area_d = mask_d.sum()
            if area_d == 0:
                continue

            intersection = np.logical_and(mask_d, mask_p).sum()
            if intersection / float(area_d) > containment_thresh:
                is_nested = True
                break

        if not is_nested:
            filtered.append(d)

    return filtered


def detect_books(image_path=None, image=None, pil_image=None, conf_threshold=CONF_THRESHOLD):
    """
    Runs SAM 3 text-prompt segmentation, returning filtered book detections.

    Accepts any combination of input:
      - image_path: path to image file (str or Path)
      - image: OpenCV BGR numpy array
      - pil_image: PIL RGB Image instance

    Returns:
        cv_image: loaded/used BGR OpenCV image (np.ndarray)
        detections: list of candidate dicts with corner points, OBB details, centroids, and masks.
    """
    if pil_image is not None and image is not None:
        cv_image = image
    elif image is not None:
        cv_image = image
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    elif image_path is not None:
        image_path = str(image_path)
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Must provide at least one of image_path, image, or pil_image to detect_books.")

    img_height, img_width = cv_image.shape[:2]
    raw_detections = []
    use_sam3 = (os.environ.get("USE_SAM3", "0") == "1")

    if not use_sam3:
        # Step [2]: Fast trained Book YOLO OBB model (~35 ms)
        yolo_mod = get_yolo_scattered()
        if yolo_mod is not None:
            results = yolo_mod(cv_image, conf=conf_threshold, verbose=False)[0]
            if results.obb is not None and len(results.obb) > 0:
                for obb in results.obb:
                    conf = float(obb.conf[0].item())
                    corners = obb.xyxyxyxy[0].cpu().numpy()

                    rect = cv2.minAreaRect(corners.astype(np.float32))
                    obb_corners = cv2.boxPoints(rect)
                    (cx, cy), (w, h), angle = rect

                    sorted_by_y = obb_corners[np.argsort(obb_corners[:, 1])]
                    top_two = sorted_by_y[:2]
                    top_edge_center_u = float(np.mean(top_two[:, 0]))
                    top_edge_center_v = float(np.mean(top_two[:, 1]))

                    raw_detections.append({
                        "conf": conf,
                        "mask_contour": corners.astype(np.int32),
                        "obb_center": (cx, cy),
                        "obb_size": (w, h),
                        "obb_angle": angle,
                        "obb_corners": obb_corners,
                        "corners": obb_corners,
                        "center": (cx, cy),
                        "width_px": float(min(w, h)),
                        "height_px": float(max(w, h)),
                        "top_edge_center": (top_edge_center_u, top_edge_center_v),
                    })

    if not raw_detections and use_sam3:
        # SAM 3 text prompt segmentation (if explicitly requested via USE_SAM3=1)
        try:
            model_sam3, processor_sam3 = get_sam3(conf_threshold)
            with torch.inference_mode():
                autocast_device = "cuda" if DEVICE == "cuda" else "cpu"
                with torch.autocast(autocast_device, dtype=torch.bfloat16):
                    state = processor_sam3.set_image(pil_image)
                    state = processor_sam3.set_text_prompt(prompt=TEXT_PROMPT, state=state)

            masks = state.get("masks", None)
            scores = state.get("scores", None)
            if masks is not None and scores is not None and len(scores) > 0:
                for i in range(len(scores)):
                    conf = float(scores[i].item())
                    if conf < conf_threshold:
                        continue
                    m = masks[i]
                    if m.dim() == 3:
                        m = m.squeeze(0)
                    mask_np = m.cpu().numpy().astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    largest = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest)
                    obb_corners = cv2.boxPoints(rect)
                    (cx, cy), (w, h), angle = rect

                    sorted_by_y = obb_corners[np.argsort(obb_corners[:, 1])]
                    top_two = sorted_by_y[:2]
                    top_edge_center_u = float(np.mean(top_two[:, 0]))
                    top_edge_center_v = float(np.mean(top_two[:, 1]))

                    raw_detections.append({
                        "conf": conf,
                        "mask_contour": largest,
                        "obb_center": (cx, cy),
                        "obb_size": (w, h),
                        "obb_angle": angle,
                        "obb_corners": obb_corners,
                        "corners": obb_corners,
                        "center": (cx, cy),
                        "width_px": float(min(w, h)),
                        "height_px": float(max(w, h)),
                        "top_edge_center": (top_edge_center_u, top_edge_center_v),
                    })
        except Exception as e:
            print(f"[Scattered] SAM 3 error: {e}", flush=True)

    filtered = filter_book_detections(raw_detections, img_width, img_height)
    return cv_image, filtered


# Alias for naming consistency with spine_pipeline.detect_spine
detect_scattered = detect_books
