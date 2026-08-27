from ultralytics import YOLO
import cv2

# ---- CONFIG ----
MODEL_PATH = r"C:\Users\joelv\runs\detect\train-6\weights\best.pt"
IMAGE_PATH = r"test\images\test_shelf_image.jpg"
OUTPUT_PATH = "result_single_book_yolo26.jpg"
CONF_THRESHOLD = 0.7

# ---- LOAD MODEL & RUN INFERENCE ----
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, imgsz=640)[0]

# ---- PICK THE SINGLE BEST DETECTION (highest confidence) ----
best_box = None
best_conf = 0

for box in results.boxes:
    conf = box.conf[0].item()
    if conf > best_conf and conf >= CONF_THRESHOLD:
        best_conf = conf
        best_box = box

if best_box is None:
    print("No book detected above confidence threshold.")
    exit()

# ---- EXTRACT VALUES ----
x1, y1, x2, y2 = best_box.xyxy[0].tolist()
width_px = x2 - x1
height_px = y2 - y1
center_u = (x1 + x2) / 2
center_v = (y1 + y2) / 2
top_edge_center_u = center_u
top_edge_center_v = y1

print("---- Book Detection Result ----")
print(f"Confidence:        {best_conf:.2f}")
print(f"Bounding box:       ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
print(f"Width:              {width_px:.1f}px")
print(f"Height:             {height_px:.1f}px")
print(f"Box center:         ({center_u:.1f}, {center_v:.1f})")
print(f"Top-edge center:    ({top_edge_center_u:.1f}, {top_edge_center_v:.1f})")

# ---- DRAW RESULT FOR VISUAL CHECK ----
img = cv2.imread(IMAGE_PATH)

cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.circle(img, (int(top_edge_center_u), int(top_edge_center_v)), 6, (0, 0, 255), -1)

label = f"w={width_px:.0f}px conf={best_conf:.2f}"
cv2.putText(img, label, (int(x1), max(int(y1) - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite(OUTPUT_PATH, img)
print(f"\nSaved visual result to: {OUTPUT_PATH}")