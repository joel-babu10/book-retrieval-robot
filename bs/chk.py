from ultralytics import YOLO
import cv2
import numpy as np

# ---- CONFIG ----
MODEL_PATH = r"C:\Users\joelv\runs\obb\train-3\weights\best.pt"  # update to your actual OBB run folder
IMAGE_PATH = r"test\images\sample.jpg"
OUTPUT_PATH = "result_single_book_obb.jpg"
CONF_THRESHOLD = 0.6

# ---- LOAD MODEL & RUN INFERENCE ----
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, imgsz=640)[0]

# ---- PICK THE SINGLE BEST DETECTION (highest confidence) ----
best_obb = None
best_conf = 0

for obb in results.obb:
    conf = obb.conf[0].item()
    if conf > best_conf and conf >= CONF_THRESHOLD:
        best_conf = conf
        best_obb = obb

if best_obb is None:
    print("No book detected above confidence threshold.")
    exit()

# ---- EXTRACT VALUES ----
# xyxyxyxy: 4 corner points of the rotated box, shape (1, 4, 2)
corners = best_obb.xyxyxyxy[0].cpu().numpy()  # shape (4, 2)

# center point = average of all 4 corners
center_u = float(np.mean(corners[:, 0]))
center_v = float(np.mean(corners[:, 1]))

# width/height from the OBB's own representation (xywhr: x, y, w, h, rotation)
xywhr = best_obb.xywhr[0].cpu().numpy()
box_w = float(xywhr[2])
box_h = float(xywhr[3])
rotation_rad = float(xywhr[4])
rotation_deg = np.degrees(rotation_rad)

# "width" of a book spine = the shorter side (spine thickness-ish in image) 
# "height" = the longer side (top-to-bottom of spine)
spine_width_px = min(box_w, box_h)
spine_length_px = max(box_w, box_h)

# top-edge center: find the midpoint of the two topmost corners (lowest y)
sorted_by_y = corners[np.argsort(corners[:, 1])]
top_two = sorted_by_y[:2]
top_edge_center_u = float(np.mean(top_two[:, 0]))
top_edge_center_v = float(np.mean(top_two[:, 1]))

print("---- Book Detection Result (OBB) ----")
print(f"Confidence:         {best_conf:.2f}")
print(f"Corners (xyxyxyxy): {corners.tolist()}")
print(f"Box center:         ({center_u:.1f}, {center_v:.1f})")
print(f"Rotation angle:      {rotation_deg:.1f} deg")
print(f"Spine width:         {spine_width_px:.1f}px")
print(f"Spine length:        {spine_length_px:.1f}px")
print(f"Top-edge center:     ({top_edge_center_u:.1f}, {top_edge_center_v:.1f})")

# ---- DRAW RESULT FOR VISUAL CHECK ----
img = cv2.imread(IMAGE_PATH)

pts = corners.astype(int).reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.circle(img, (int(top_edge_center_u), int(top_edge_center_v)), 6, (0, 0, 255), -1)

label = f"w={spine_width_px:.0f}px angle={rotation_deg:.0f} conf={best_conf:.2f}"
cv2.putText(img, label, (int(center_u) - 60, int(center_v)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(OUTPUT_PATH, img)
print(f"\nSaved visual result to: {OUTPUT_PATH}")