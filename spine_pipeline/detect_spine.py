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
CONF_THRESHOLD = 0.12

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


def _suppress_overlapping_spines(candidates, overlap_thresh=0.45):
    """
    Suppresses redundant overlapping detections on the same physical book spine.
    For vertical shelf spines, checks 1D horizontal overlap and 2D bounding overlap.
    """
    if len(candidates) <= 1:
        return candidates

    kept = []
    for cand in candidates:
        cand_corners = cand["corners"]
        cand_x_min = float(np.min(cand_corners[:, 0]))
        cand_x_max = float(np.max(cand_corners[:, 0]))
        cand_y_min = float(np.min(cand_corners[:, 1]))
        cand_y_max = float(np.max(cand_corners[:, 1]))
        cand_w = max(cand_x_max - cand_x_min, 1.0)
        cand_h = max(cand_y_max - cand_y_min, 1.0)

        is_duplicate = False
        for k in kept:
            k_corners = k["corners"]
            k_x_min = float(np.min(k_corners[:, 0]))
            k_x_max = float(np.max(k_corners[:, 0]))
            k_y_min = float(np.min(k_corners[:, 1]))
            k_y_max = float(np.max(k_corners[:, 1]))
            k_w = max(k_x_max - k_x_min, 1.0)
            k_h = max(k_y_max - k_y_min, 1.0)

            x_inter = max(0.0, min(cand_x_max, k_x_max) - max(cand_x_min, k_x_min))
            x_overlap = x_inter / min(cand_w, k_w)

            y_inter = max(0.0, min(cand_y_max, k_y_max) - max(cand_y_min, k_y_min))
            y_overlap = y_inter / min(cand_h, k_h)

            if x_overlap >= overlap_thresh and y_overlap >= 0.25:
                # If the new candidate is significantly wider and complete (e.g. full spine vs crease sliver),
                # upgrade the kept candidate to the full spine
                if cand_w > 1.30 * k_w and cand["conf"] >= 0.40:
                    kept[kept.index(k)] = cand
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(cand)

    return kept


BOOK_MODEL_PATH = r"C:\Users\joelv\runs\obb\runs\obb\scattered-train-2-3\weights\best.pt"
_book_model = None


def get_book_model():
    """Lazy-load the full book YOLO OBB model for detecting wide/end-of-shelf upright books."""
    global _book_model
    if _book_model is None and Path(BOOK_MODEL_PATH).exists():
        _book_model = YOLO(BOOK_MODEL_PATH)
    return _book_model


def detect_spines(image_path=None, image=None, conf_threshold=CONF_THRESHOLD, model_path=None, refine_sam=False):
    """
    Runs YOLO OBB spine detection followed by SAM mask edge refinement.
    Detects both narrow shelf spines and wide upright shelf books (e.g. end-of-shelf books).

    Workflow:
      Stage 1: YOLO OBB Detection -> generates candidate oriented bounding boxes.
      Stage 2: Geometric & NMS Filtering -> filters out tiny noise & duplicate columns.
      Stage 3: SAM Mask Refinement -> refines bounding box contour edges (if requested).
      Stage 4: Feature Extraction  -> computes centroids, spine width, and top-edge points.

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
    if isinstance(image_path, np.ndarray):
        image = image_path
        image_path = None

    model = get_model(model_path)

    if image is not None:
        img = image
        results = model(img, imgsz=640, conf=conf_threshold, iou=0.45, device=DEVICE, verbose=False)[0]
    else:
        results = model(image_path, imgsz=640, conf=conf_threshold, iou=0.45, device=DEVICE, verbose=False)[0]
        import cv2
        img = cv2.imread(image_path)

    raw_candidates = []

    # 1a. Gather raw YOLO OBB candidates from spine model with geometric spine validation
    if results.obb is not None and len(results.obb) > 0:
        for obb in results.obb:
            conf = obb.conf[0].item()
            if conf < conf_threshold:
                continue
            corners = obb.xyxyxyxy[0].cpu().numpy()
            xywhr = obb.xywhr[0].cpu().numpy()
            img_h, img_w = img.shape[:2]
            cx, cy = float(xywhr[0]), float(xywhr[1])
            if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
                continue

            w_ext = float(np.ptp(corners[:, 0]))
            h_ext = float(np.ptp(corners[:, 1]))
            # Shelf book spines stand upright vertically (height > width)
            # Rejects horizontal wall seams, table borders, and floor shadows
            if h_ext < 1.20 * w_ext or h_ext < 120.0 or w_ext < 10.0:
                continue

            raw_candidates.append({
                "corners": corners,
                "xywhr": xywhr,
                "conf": conf,
            })

    # 1b. Gather upright shelf books from book model and extract the physical spine strip
    b_model = get_book_model()
    if b_model is not None:
        try:
            b_results = b_model(img, imgsz=640, conf=0.35, iou=0.45, device=DEVICE, verbose=False)[0]
            if b_results.obb is not None and len(b_results.obb) > 0:
                for obb in b_results.obb:
                    conf = obb.conf[0].item()
                    corners = obb.xyxyxyxy[0].cpu().numpy()
                    b_xywhr = obb.xywhr[0].cpu().numpy()
                    h = float(max(b_xywhr[2], b_xywhr[3]))
                    angle_rad = float(b_xywhr[4])
                    is_upright = abs(angle_rad) < 0.45 or abs(angle_rad - np.pi) < 0.45 or abs(angle_rad + np.pi) < 0.45
                    if is_upright and h >= 220.0:
                        sorted_by_y = corners[np.argsort(corners[:, 1])]
                        top_two = sorted_by_y[:2]
                        bot_two = sorted_by_y[2:]
                        top_two = top_two[np.argsort(top_two[:, 0])]
                        bot_two = bot_two[np.argsort(bot_two[:, 0])]
                        tl, tr = top_two[0], top_two[1]
                        bl, br = bot_two[0], bot_two[1]

                        book_w = float(np.linalg.norm(tr - tl))
                        aspect_ratio = h / max(book_w, 1.0)

                        # If book_w is a typical spine width (<= 180px) or aspect ratio >= 2.2:
                        # The upright book detection IS the full book spine face itself!
                        # Keep full width so OCR reads all text and Dobot targets true spine center.
                        if book_w <= 180.0 or aspect_ratio >= 2.2:
                            raw_candidates.append({
                                "corners": corners,
                                "xywhr": b_xywhr,
                                "conf": conf,
                            })
                        else:
                            # Wide book showing front cover (e.g. end-of-shelf display): slice the edge spine
                            spine_w = min(115.0, max(35.0, book_w * 0.35))
                            dir_top = (tr - tl) / max(book_w, 1e-4)
                            dir_bot = (br - bl) / max(float(np.linalg.norm(br - bl)), 1e-4)

                            sp_tl = tl
                            sp_tr = tl + spine_w * dir_top
                            sp_br = bl + spine_w * dir_bot
                            sp_bl = bl
                            sp_corners = np.array([sp_tl, sp_tr, sp_br, sp_bl], dtype=np.float32)
                            sp_center_x = float(np.mean(sp_corners[:, 0]))
                            sp_center_y = float(np.mean(sp_corners[:, 1]))
                            sp_xywhr = np.array([sp_center_x, sp_center_y, spine_w, h, angle_rad], dtype=np.float32)

                            raw_candidates.append({
                                "corners": sp_corners,
                                "xywhr": sp_xywhr,
                                "conf": conf,
                            })

                            if book_w > 200.0:
                                sp_tr2 = tr
                                sp_tl2 = tr - spine_w * dir_top
                                sp_br2 = br
                                sp_bl2 = br - spine_w * dir_bot
                                sp_corners2 = np.array([sp_tl2, sp_tr2, sp_br2, sp_bl2], dtype=np.float32)
                                sp_cx2 = float(np.mean(sp_corners2[:, 0]))
                                sp_cy2 = float(np.mean(sp_corners2[:, 1]))
                                sp_xywhr2 = np.array([sp_cx2, sp_cy2, spine_w, h, angle_rad], dtype=np.float32)
                                raw_candidates.append({
                                    "corners": sp_corners2,
                                    "xywhr": sp_xywhr2,
                                    "conf": conf * 0.95,
                                })
        except Exception:
            pass

    if not raw_candidates:
        return img, []

    # Sort by confidence descending
    raw_candidates.sort(key=lambda x: x["conf"], reverse=True)

    # Suppress redundant column duplicates on the same spine
    raw_candidates = _suppress_overlapping_spines(raw_candidates, overlap_thresh=0.45)

    # Limit to top realistic candidates (up to 25 books across shelf frame)
    raw_candidates = raw_candidates[:25]

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
        top_two = top_two[np.argsort(top_two[:, 0])]
        tl, tr = top_two[0], top_two[1]

        top_edge_center_u = float((tl[0] + tr[0]) / 2.0)
        top_edge_center_v = float((tl[1] + tr[1]) / 2.0)
        pick_endpoints = [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1]))]

        # Inset pick point slightly down into spine body away from background wall
        v_in_u = center_u - top_edge_center_u
        v_in_v = center_v - top_edge_center_v
        dist_to_center = float(np.hypot(v_in_u, v_in_v))
        inset_px = min(35.0, max(15.0, dist_to_center * 0.08)) if dist_to_center > 0 else 0.0
        pick_u = top_edge_center_u + (v_in_u / max(dist_to_center, 1e-4)) * inset_px
        pick_v = top_edge_center_v + (v_in_v / max(dist_to_center, 1e-4)) * inset_px

        detections.append({
            "corners": corners,
            "xywhr": xywhr,
            "conf": conf,
            "center": (center_u, center_v),
            "width_px": spine_width_px,
            "top_edge_center": (pick_u, pick_v),
            "pick_edge_endpoints": pick_endpoints,
        })

    return img, detections
