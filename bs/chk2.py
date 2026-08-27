from ultralytics import YOLO
import cv2
import numpy as np

# ---- CONFIG ----
MODEL_PATH = r"C:\Users\joelv\runs\obb\train-3\weights\best.pt"
IMAGE_PATH = r"test\images\sample.jpg"
OUTPUT_PATH = "result_all_books_obb.jpg"
CONF_THRESHOLD = 0.4
IOU_OVERLAP_THRESHOLD = 0.4  # duplicate suppression

def rotated_iou_approx(c1, c2):
    """Rough overlap check using axis-aligned bounding rect of each rotated box."""
    def to_aabb(corners):
        xs = corners[:, 0]
        ys = corners[:, 1]
        return xs.min(), ys.min(), xs.max(), ys.max()

    x1a, y1a, x2a, y2a = to_aabb(c1)
    x1b, y1b, x2b, y2b = to_aabb(c2)

    xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
    xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

# ---- LOAD MODEL & RUN INFERENCE ----
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded. Running inference...")
results = model(IMAGE_PATH, imgsz=640)[0]
print("Inference done.")

# ---- COLLECT DETECTIONS ABOVE THRESHOLD ----
detections = []
for obb in results.obb:
    conf = obb.conf[0].item()
    if conf < CONF_THRESHOLD:
        continue
    corners = obb.xyxyxyxy[0].cpu().numpy()
    xywhr = obb.xywhr[0].cpu().numpy()
    detections.append({
        "corners": corners,
        "conf": conf,
        "w": float(xywhr[2]),
        "h": float(xywhr[3]),
        "angle": float(np.degrees(xywhr[4]))
    })

# sort by confidence, remove overlapping duplicates
detections.sort(key=lambda d: d["conf"], reverse=True)
filtered = []
for det in detections:
    overlap_found = False
    for kept in filtered:
        if rotated_iou_approx(det["corners"], kept["corners"]) > IOU_OVERLAP_THRESHOLD:
            overlap_found = True
            break
    if not overlap_found:
        filtered.append(det)

print(f"Before filtering: {len(detections)} detections")
print(f"After filtering:  {len(filtered)} detections\n")

# ---- PRINT + DRAW EACH BOOK ----
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"ERROR: could not load image at {IMAGE_PATH}")
    exit()

for i, det in enumerate(filtered):
    corners = det["corners"]
    conf = det["conf"]

    center_u = float(np.mean(corners[:, 0]))
    center_v = float(np.mean(corners[:, 1]))

    spine_width_px = min(det["w"], det["h"])
    spine_length_px = max(det["w"], det["h"])

    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    top_edge_center_u = float(np.mean(top_two[:, 0]))
    top_edge_center_v = float(np.mean(top_two[:, 1]))

    print(f"Book {i+1}: conf={conf:.2f}  width={spine_width_px:.0f}px  "
          f"length={spine_length_px:.0f}px  angle={det['angle']:.1f}deg  "
          f"center=({center_u:.1f},{center_v:.1f})  "
          f"top_edge_center=({top_edge_center_u:.1f},{top_edge_center_v:.1f})")

    pts = corners.astype(int).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.circle(img, (int(top_edge_center_u), int(top_edge_center_v)), 4, (0, 0, 255), -1)
    cv2.putText(img, f"{i+1}", (int(center_u) - 5, int(center_v)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imwrite(OUTPUT_PATH, img)
print(f"\nSaved visual result to: {OUTPUT_PATH}")