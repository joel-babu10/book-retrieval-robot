"""Operator UI backend. Run: pip install flask opencv-python numpy && python server.py  ->  http://localhost:5000

Hook your pipeline in pipeline.py (same folder) exposing:
  find_target(query:str, mode:str) -> dict | None
      {"title", "conf", "sim", "mode", "cam_xyz":[x,y,z] (m), "robot_xyz":[x,y,z] or None,
       "pick_note", "frame": annotated BGR ndarray (optional)}
  get_frame() -> annotated BGR ndarray (live view)
  send_to_arm(robot_xyz) -> bool
Missing pipeline.py -> demo data so the UI still runs.
"""
import time, threading
from pathlib import Path
import cv2, numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from pipeline import find_target, get_frame, send_to_arm
    DEMO = False
except ImportError:
    DEMO = True

    def get_frame():
        img = np.full((484, 860, 3), (37, 29, 22), np.uint8)
        cv2.putText(img, "DEMO FEED (pipeline.py not found)", (230, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 190, 200), 2)
        return img

    def find_target(query, mode):
        time.sleep(0.6)
        return {"title": query, "conf": 0.89, "sim": 1.0, "mode": "spine" if mode == "auto" else mode,
                "cam_xyz": [0.090, -0.153, 0.419], "robot_xyz": None,
                "pick_note": "Upper spine, centred on width"}

    def send_to_arm(robot_xyz):
        return False

# Ensure static folder works whether running from root or elsewhere
STATIC_DIR = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
_lock = threading.Lock()
_last = {"result": None}


@app.get("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.get("/<path:filename>")
def static_file(filename):
    return send_from_directory(str(STATIC_DIR), filename)


def _mjpeg():
    while True:
        frame = get_frame()
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1 / 15)


@app.get("/video_feed")
def video_feed():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.post("/api/find")
def api_find():
    d = request.get_json(force=True)
    query, mode = (d.get("query") or "").strip(), d.get("mode", "auto")
    if not query:
        return jsonify(error="empty query"), 400
    with _lock:
        res = find_target(query, mode)
        if res is None:
            return jsonify(found=False)
        res = {k: v for k, v in res.items() if k != "frame"}
        _last["result"] = res
    return jsonify(found=True, **res)


@app.post("/api/send")
def api_send():
    res = _last["result"]
    if not res or res.get("robot_xyz") is None:
        return jsonify(ok=False, error="no robot-frame target"), 400
    return jsonify(ok=bool(send_to_arm(res["robot_xyz"])))


if __name__ == "__main__":
    print(f"[Operator UI] Server starting at http://localhost:5000 (DEMO={DEMO})", flush=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)
