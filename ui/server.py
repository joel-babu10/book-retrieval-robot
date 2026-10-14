"""
ui/server.py
----------------------------------------------------------------------------
Flask Web API Server for the Autonomous Book Retrieval Robot UI.
Exposes endpoints for automated mode selection, spine/scattered/mixed target
detection, SAM geometric refinement, and Dobot Magician trajectory execution.
"""

import os
import sys
import json
import base64
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, jsonify, request, send_from_directory
import torch

from shared.mode_classifier import classify_mode
from spine_pipeline.find_target_spine import find_target_spine_image, find_target_spine_live
from scattered_pipeline.find_target_scattered import find_target_scattered_image, find_target_scattered_live
from mixed_pipeline.find_target_mixed import find_target_mixed_image, find_target_mixed_live

app = Flask(__name__, static_folder="static", static_url_path="")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_IMAGES = {
    "shelf_spine": {
        "title": "Shelf Spine (JavaServer Pages)",
        "query": "JavaServer Pages",
        "path": r"C:\Users\joelv\.gemini\antigravity-ide\brain\007729dc-2ebd-4814-b85d-052dceb435cf\.user_uploaded\media_1791955764515.jpg"
    },
    "stacked_mixed": {
        "title": "Stacked / Wall Desk (Concepts of Database Management)",
        "query": "Database Management",
        "path": r"C:\Users\joelv\.gemini\antigravity-ide\brain\007729dc-2ebd-4814-b85d-052dceb435cf\.user_uploaded\media_1791955774876.jpg"
    },
    "scattered_flat": {
        "title": "Scattered Flat (JavaServer Pages)",
        "query": "JavaServer Pages",
        "path": r"C:\Users\joelv\.gemini\antigravity-ide\brain\007729dc-2ebd-4814-b85d-052dceb435cf\.user_uploaded\media_1791955785260.jpg"
    }
}


def image_to_base64(img_bgr):
    """Encodes a BGR cv2 image to a base64 JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status", methods=["GET"])
def get_system_status():
    camera_ready = False
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        camera_ready = len(ctx.query_devices()) > 0
    except Exception:
        camera_ready = False

    dobot_ports = []
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            dobot_ports.append({"device": p.device, "description": p.description})
    except Exception:
        pass

    catalog = []
    try:
        from shared.ocr_reader import _get_catalog
        catalog = [b.get("title") for b in _get_catalog() if "title" in b]
    except Exception:
        catalog = ["JavaServer Pages", "Database Management", "Linear Algebra", "Probability & Random Processes"]

    return jsonify({
        "status": "online",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "camera_connected": camera_ready,
        "dobot_ports": dobot_ports,
        "catalog": catalog,
        "samples": {k: {"title": v["title"], "query": v["query"]} for k, v in SAMPLE_IMAGES.items()}
    })


@app.route("/api/detect", methods=["POST"])
def run_detection():
    data = request.json or {}
    source = data.get("source", "sample")  # 'sample' or 'live'
    sample_key = data.get("sample_key", "shelf_spine")
    query_title = data.get("query", "JavaServer Pages")
    requested_mode = data.get("mode", "auto")

    # 1. Acquire Image
    cv_img = None
    depth_frame = None
    intrinsics = None

    if source == "live":
        try:
            from input.camera import RealSenseCamera
            with RealSenseCamera() as cam:
                cv_img, depth_frame, intrinsics = cam.capture()
            if cv_img is None:
                return jsonify({"error": "Failed to capture frame from RealSense camera."}), 500
        except Exception as e:
            return jsonify({"error": f"Camera capture failed: {e}"}), 500
    else:
        sample_info = SAMPLE_IMAGES.get(sample_key, SAMPLE_IMAGES["shelf_spine"])
        img_path = sample_info["path"]
        if not os.path.exists(img_path):
            return jsonify({"error": f"Sample image file not found: {img_path}"}), 404
        cv_img = cv2.imread(img_path)

    raw_base64 = image_to_base64(cv_img)

    # 2. Automated Mode Selection (if auto)
    classified_mode, mode_conf = classify_mode(cv_img, verbose=False)
    active_mode = classified_mode if requested_mode == "auto" else requested_mode

    # 3. Pipeline Execution
    result = None
    output_temp = ROOT_DIR / "ui" / "static" / "temp_result.jpg"
    output_temp.parent.mkdir(parents=True, exist_ok=True)

    if source == "live":
        if active_mode == "spine":
            result = find_target_spine_live(
                target_title=query_title,
                cv_image=cv_img,
                depth_frame=depth_frame,
                color_intrinsics=intrinsics,
                output_path=str(output_temp),
                refine_sam=True
            )
        elif active_mode == "scattered":
            result = find_target_scattered_live(
                query_title=query_title,
                cv_image=cv_img,
                depth_frame=depth_frame,
                color_intrinsics=intrinsics,
                output_path=str(output_temp),
                refine_sam=True
            )
        else:
            result = find_target_mixed_live(
                query_title=query_title,
                cv_image=cv_img,
                depth_frame=depth_frame,
                color_intrinsics=intrinsics,
                output_path=str(output_temp),
                refine_sam=True
            )
    else:
        sample_path = SAMPLE_IMAGES.get(sample_key, {}).get("path")
        if active_mode == "spine":
            result = find_target_spine_image(
                image_path=sample_path,
                target_title=query_title,
                output_path=str(output_temp),
                refine_sam=True
            )
        elif active_mode == "scattered":
            result = find_target_scattered_image(
                image_path=sample_path,
                query_title=query_title,
                output_path=str(output_temp),
                refine_sam=True
            )
        else:
            result = find_target_mixed_image(
                image_path=sample_path,
                query_title=query_title,
                output_path=str(output_temp),
                refine_sam=True
            )

    annotated_base64 = ""
    if output_temp.exists():
        ann_img = cv2.imread(str(output_temp))
        if ann_img is not None:
            annotated_base64 = image_to_base64(ann_img)

    # Clean serializable payload
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    clean_result = sanitize(result) if result else None

    return jsonify({
        "success": clean_result is not None,
        "mode": active_mode,
        "mode_conf": float(mode_conf),
        "target": clean_result,
        "raw_image": raw_base64,
        "annotated_image": annotated_base64
    })


@app.route("/api/pick", methods=["POST"])
def execute_robot_pick():
    data = request.json or {}
    xyz = data.get("xyz", [-0.05, 0.0, 0.50])
    angle = data.get("angle", 0.0)
    is_spine = data.get("is_spine", False)

    from robot.dobot_controller import get_dobot_controller
    controller = get_dobot_controller()
    pick_result = controller.execute_pick_sequence(xyz, grasp_angle_deg=angle, is_spine=is_spine)

    return jsonify(pick_result)


if __name__ == "__main__":
    print("[UI Server] Starting Autonomous Robot UI on http://localhost:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
