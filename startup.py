"""
startup.py
----------------------------------------------------------------------------
Overlaps startup costs concurrently:
  1. Camera warmup and services (Shared EasyOCR, SAM Refine, Mode Classifier)
     preloading quietly in background threads.
  2. Concurrently capturing the user's book query (voice / text).

The automatic mode classifier replaces manual mode selection, determining
whether books are stacked spines or scattered covers directly from the
captured camera frame in ~2ms.
"""

import threading

from input.query_input import get_book_query
from input.camera import RealSenseCamera
from shared.ocr_reader import preload_shared_ocr_reader
from shared.sam_refine import preload_sam_refine
from shared.mode_classifier import preload_mode_classifier


from spine_pipeline.detect_spine import preload_spine_model
from scattered_pipeline.detect_scattered import preload_scattered_model


def parallel_startup():
    """
    Initializes camera warmup, detection models, and shared services in
    background threads while concurrently capturing the user's book query.
    Returns:
        query_result (dict or None), camera (RealSenseCamera instance)
    """
    result_holder = {}
    camera = RealSenseCamera()

    camera_thread = threading.Thread(target=camera.start, kwargs={"verbose": False}, daemon=True)
    ocr_thread = threading.Thread(target=preload_shared_ocr_reader, kwargs={"verbose": False}, daemon=True)
    refine_thread = threading.Thread(target=preload_sam_refine, kwargs={"verbose": False}, daemon=True)
    cls_thread = threading.Thread(target=preload_mode_classifier, kwargs={"verbose": False}, daemon=True)
    spine_thread = threading.Thread(target=preload_spine_model, kwargs={"verbose": False}, daemon=True)
    scattered_thread = threading.Thread(target=preload_scattered_model, kwargs={"verbose": False}, daemon=True)

    def _run_query():
        result_holder["query_result"] = get_book_query()

    query_thread = threading.Thread(target=_run_query, daemon=True)

    print("[Startup] Initializing camera, OCR & services in background...", flush=True)

    camera_thread.start()
    ocr_thread.start()
    refine_thread.start()
    cls_thread.start()
    spine_thread.start()
    scattered_thread.start()
    query_thread.start()

    # Wait for user query input
    query_thread.join()

    # Finalize background threads
    ocr_thread.join()
    refine_thread.join()
    cls_thread.join()
    spine_thread.join()
    scattered_thread.join()
    camera_thread.join()

    print("[Startup] System ready.\n", flush=True)
    return result_holder.get("query_result"), camera


if __name__ == "__main__":
    result, cam = parallel_startup()
    if result is None:
        print("No query captured.")
    else:
        print(f"Query matched: {result['title']}")
        cv_image, depth_frame, intrinsics = cam.capture()
        print("Camera frame captured successfully.")
    cam.stop()
