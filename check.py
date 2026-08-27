from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\Users\joelv\runs\detect\train-7\weights\best.pt"
IMAGE_PATH = r"test\images\sample.jpg"
OUTPUT_PATH = "result_all_boxes_yolo26.jpg"
IOU_OVERLAP_THRESHOLD = 0.4  # boxes overlapping more than this = treated as duplicates
CONF_THRESHOLD = 0.6  # drop weak/spurious detections (e.g. barcode labels, edge noise)

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, imgsz=640)[0]

# collect all detections above confidence threshold
detections = []
for box in results.boxes:
    conf = box.conf[0].item()
    if conf < CONF_THRESHOLD:
        continue
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    detections.append({"box": (x1, y1, x2, y2), "conf": conf})

# sort by confidence, highest first
detections.sort(key=lambda d: d["conf"], reverse=True)

# remove overlapping duplicates, keep highest-confidence version
filtered = []
for det in detections:
    overlap_found = False
    for kept in filtered:
        if box_iou(det["box"], kept["box"]) > IOU_OVERLAP_THRESHOLD:
            overlap_found = True
            break
    if not overlap_found:
        filtered.append(det)

print(f"\nBefore filtering: {len(detections)} detections")
print(f"After filtering:  {len(filtered)} detections\n")

img = cv2.imread(IMAGE_PATH)

for det in filtered:
    x1, y1, x2, y2 = det["box"]
    conf = det["conf"]
    print(f"conf={conf:.2f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  width={x2-x1:.0f}")

    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(img, f"{conf:.2f}", (x1i, max(y1i - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(OUTPUT_PATH, img)
print(f"\nSaved visual result to: {OUTPUT_PATH}")