"""Operator UI backend. Run: python server.py -> http://localhost:5000"""

import os, sys, subprocess, logging

from pathlib import Path



# Configure Windows Console and stdout to prevent freezing/blocking

def _setup_console():

    if hasattr(sys.stdout, "reconfigure"):

        try:

            sys.stdout.reconfigure(line_buffering=True, write_through=True)

            sys.stderr.reconfigure(line_buffering=True, write_through=True)

        except Exception:

            pass

    if os.name == "nt":

        try:

            import ctypes

            kernel32 = ctypes.windll.kernel32

            # Disable QuickEdit Mode (0x0040) so clicking inside the terminal does not pause process execution

            hStdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10

            mode = ctypes.c_ulong()

            if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):

                new_mode = (mode.value & ~0x0040) | 0x0080

                kernel32.SetConsoleMode(hStdin, new_mode)

        except Exception:

            pass



_setup_console()



# Silence noisy Werkzeug request access logs so they never block on console writes

logging.getLogger("werkzeug").setLevel(logging.WARNING)



# Auto-relaunch inside GPU-enabled virtual environment if run via CPU interpreter

_ROOT = Path(__file__).resolve().parent

_VENV_PY = _ROOT / "venv" / "Scripts" / "python.exe"

if __name__ == "__main__" and _VENV_PY.exists():

    try:

        _curr = Path(sys.executable).resolve()

        _target = _VENV_PY.resolve()

        if _curr != _target:

            import torch

            _has_cuda = torch.cuda.is_available()

        else:

            _has_cuda = True

    except Exception:

        _has_cuda = False



    if not _has_cuda:

        try:

            env = os.environ.copy()

            env["PYTHONUNBUFFERED"] = "1"

            res = subprocess.run([str(_VENV_PY), "-u"] + sys.argv, env=env)

            sys.exit(res.returncode)

        except KeyboardInterrupt:

            sys.exit(0)



if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))

_THIS_DIR = str(Path(__file__).resolve().parent)

if _THIS_DIR not in sys.path:

    sys.path.insert(0, _THIS_DIR)



import time, threading

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

    try:

        while True:

            frame = get_frame()

            if frame is not None:

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                if ok:

                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

            time.sleep(1 / 20)

    except GeneratorExit:

        pass

    except Exception:

        pass





@app.get("/video_feed")

def video_feed():

    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=f")





@app.get("/api/status")
def api_status():
    try:
        from pipeline import _latest_color, _camera_active, _current_target
        return jsonify(
            camera_live=_latest_color is not None,
            camera_active=bool(_camera_active),
            target_locked=_current_target is not None,
            result_ready=_last["result"] is not None
        )
    except Exception:
        return jsonify(camera_live=True, camera_active=True, target_locked=False, result_ready=False)


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





@app.post("/api/clear")

def api_clear():

    with _lock:

        _last["result"] = None

        try:

            from pipeline import clear_target

            clear_target()

        except Exception:

            pass

    return jsonify(ok=True)





@app.post("/api/send")

def api_send():

    res = _last["result"]

    if not res or res.get("robot_xyz") is None:

        return jsonify(ok=False, error="no robot-frame target"), 400

    return jsonify(ok=bool(send_to_arm(res["robot_xyz"])))





if __name__ == "__main__":

    print(f"[Operator UI] Server starting at http://localhost:5000 (DEMO={DEMO})", flush=True)

    app.run(host="0.0.0.0", port=5000, threaded=True)

