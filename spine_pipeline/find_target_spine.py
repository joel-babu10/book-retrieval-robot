"""

spine_pipeline/find_target_spine.py

----------------------------------------------------------------------------

Combines detect_spine.py (YOLO OBB) + ocr_match_spine.py (OCR + title

matching) into the full spine-finding flow. Supports:



  - find_target_spine_image(...)  — run on a saved image file (testing)

  - find_target_spine_live(...)   — run on a live RealSense frame,

                                     including depth deprojection to XYZ

                                     (uses input/camera.py + input/depth_utils.py)

"""



import sys

from pathlib import Path



# Ensure project root is in sys.path when running script directly or as package

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(0, str(ROOT_DIR))



import cv2

import numpy as np



from input.query_input import get_book_query

from spine_pipeline.detect_spine import detect_spines, CONF_THRESHOLD as DEFAULT_CONF

from spine_pipeline.ocr_match_spine import ocr_spine_crop, get_reader





def _pick_best_match(img, detections, target_title, reader=None, min_similarity=0.35):

    """Runs OCR matching on every detection, returns the best-scoring one

    (with 'text' and 'similarity' added), or None if there were no

    detections or if best match is below min_similarity."""

    if not detections:

        print("\n[Spine] No detections above threshold.", flush=True)

        return None



    reader = reader or get_reader()



    sorted_detections = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)

    best = None



    for d in sorted_detections:

        text, similarity = ocr_spine_crop(img, d["corners"], target_title, reader=reader)

        d["text"] = text

        d["similarity"] = similarity

        print(f"  conf={d['conf']:.2f}  OCR='{text}'  match={similarity:.2f}", flush=True)



        if best is None or similarity > best["similarity"]:

            best = d



        # Confident match — avoid spending time OCRing remaining candidates

        if similarity >= 0.85:

            print(f"  [Spine Match] Confident match ({similarity:.2f} >= 0.85). Early exit!", flush=True)

            break



    if best is None or best.get("similarity", 0.0) < min_similarity:

        sim = best.get("similarity", 0.0) if best is not None else 0.0

        print(f"\n[Spine] Best match similarity ({sim:.2f}) is below threshold ({min_similarity:.2f}). No confident target found.", flush=True)

        return None



    print("\n---- Best Match ----", flush=True)

    print(f"OCR text:           '{best['text']}'", flush=True)

    print(f"Similarity score:    {best['similarity']:.2f}", flush=True)

    print(f"Center (u, v):       ({best['center'][0]:.1f}, {best['center'][1]:.1f})", flush=True)

    print(f"Width:               {best['width_px']:.1f}px", flush=True)

    print(f"Top-edge center:     ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)



    return best





def _save_visual_result(img, best, output_path, xyz=None):
    if best is None or img is None:
        return img

    from shared.display_utils import draw_aesthetic_target_overlay
    res = draw_aesthetic_target_overlay(img, best)

    try:
        cv2.imwrite(output_path, res)
        print(f"\nSaved aesthetic visual result to: {output_path}", flush=True)

        from datetime import datetime
        runs_dir = Path(output_path).parent / "saved_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(output_path).stem
        history_path = runs_dir / f"{stem}_{timestamp}.jpg"
        cv2.imwrite(str(history_path), res)
    except Exception as e:
        print(f"Notice saving visual result: {e}", flush=True)

    return res


def _refine_target_with_sam(img, best):

    """

    Runs SAM mask refinement exclusively on the single matched target book.

    Updates best["corners"], best["center"], best["top_edge_center"], and best["width_px"]

    to exact SAM-refined coordinates.

    """

    if best is None or "corners" not in best:

        return best



    try:

        from shared.sam_refine import refine_obb_with_sam

        print(f"\n[SAM Refine] Target book matched ('{best['text']}'). Running SAM refinement exclusively on target...", flush=True)

        res = refine_obb_with_sam(img, best["corners"], return_contour=True)

        if isinstance(res, tuple):

            refined_corners, sam_contour = res

        else:

            refined_corners, sam_contour = res, None



        if refined_corners is not None:

            best["corners"] = refined_corners

            if sam_contour is not None:

                best["mask_contour"] = sam_contour



            from shared.sam_refine import extract_refined_pick_edge

            edge_info = extract_refined_pick_edge(refined_corners, pick_location="top_center")

            pick_u, pick_v = edge_info["pick_point_2d"]

            endpoints = edge_info["pick_edge_endpoints"]

            edge_len = edge_info["pick_edge_length_px"]

            center_x, center_y = edge_info["center_2d"]



            best["pick_point_2d"] = (pick_u, pick_v)

            best["top_edge_center"] = (pick_u, pick_v)

            best["pick_point_px"] = [pick_u, pick_v]

            best["pick_edge_endpoints"] = endpoints

            best["pick_edge_angle_deg"] = edge_info["pick_edge_angle_deg"]

            best["grasp_approach_angle_deg"] = edge_info["grasp_approach_angle_deg"]

            best["is_short_edge"] = edge_info["is_short_edge"]

            best["pick_edge_length_px"] = edge_len

            best["obb_angle"] = edge_info["pick_edge_angle_deg"]

            best["center"] = (center_x, center_y)



            xywhr = best.get("xywhr")

            if xywhr is not None:

                best["width_px"] = float(edge_len)



            print(f"[SAM Refine] Target refined exact coordinates -> Center: ({best['center'][0]:.1f}, {best['center'][1]:.1f}), Top-Edge Center Pick: ({best['top_edge_center'][0]:.1f}, {best['top_edge_center'][1]:.1f})", flush=True)

    except Exception as e:

        print(f"[SAM Refine] Target refinement fallback to YOLO OBB: {e}", flush=True)



    return best





# =========================================================

# IMAGE MODE — test on a saved file, no camera/depth needed

# =========================================================



def find_target_spine_image(image_path, target_title, conf_threshold=DEFAULT_CONF,

                             save_visual=True, output_path="result_target_spine.jpg",

                             refine_sam=True):

    """

    Runs spine detection (YOLO OBB) + OCR matching on a saved image file,

    then runs SAM edge refinement EXCLUSIVELY on the detected target book.



    Returns a target_output dict (u, v, width_px, top-edge center, etc.)

    or None if nothing matched.

    """

    img, detections = detect_spines(image_path=image_path, conf_threshold=conf_threshold, refine_sam=False)

    print(f"[Spine] {len(detections)} spine candidate(s) detected.", flush=True)



    best = _pick_best_match(img, detections, target_title)



    if best is not None:

        if refine_sam:

            best = _refine_target_with_sam(img, best)

        if save_visual:

            _save_visual_result(img, best, output_path)



    if best is None:

        return None



    target_output = {

        "u": best["center"][0],

        "v": best["center"][1],

        "width_px": best["width_px"],

        "top_edge_center_u": best["top_edge_center"][0],

        "top_edge_center_v": best["top_edge_center"][1],

        "pick_point_2d": best.get("pick_point_2d", best["top_edge_center"]),

        "pick_point_px": [best["top_edge_center"][0], best["top_edge_center"][1]],

        "pick_edge_endpoints": best.get("pick_edge_endpoints"),

        "pick_edge_angle_deg": best.get("pick_edge_angle_deg"),

        "grasp_approach_angle_deg": best.get("grasp_approach_angle_deg"),

        "is_short_edge": best.get("is_short_edge"),

        "pick_edge_length_px": best.get("pick_edge_length_px"),

        "similarity": best["similarity"],

        "conf": best["conf"],

        "text": best["text"],

        "corners": best.get("corners"),

        "mask_contour": best.get("mask_contour"),

    }

    from shared.display_utils import print_target_summary

    print_target_summary(target_output)



    return target_output





# =========================================================

# LIVE MODE — RealSense capture + depth deprojection

# =========================================================



def find_target_spine_live(target_title, conf_threshold=DEFAULT_CONF,

                            save_visual=True, output_path="result_target_spine_live.jpg",

                            camera=None, cv_image=None, depth_frame=None, color_intrinsics=None,

                            refine_sam=True):

    """

    Full live pipeline: captures an aligned color+depth frame from the

    RealSense camera, runs spine detection + OCR matching to identify the

    target book, then runs SAM refinement EXCLUSIVELY on the target book box

    to get exact pixel coordinates before deprojecting to real-world XYZ.

    """

    from input.camera import RealSenseCamera

    from input.depth_utils import deproject_target



    have_external_frame = (cv_image is not None and depth_frame is not None and color_intrinsics is not None)

    owns_camera = (camera is None and not have_external_frame)



    if owns_camera:

        camera = RealSenseCamera()

        camera.start()

    elif camera is not None and not getattr(camera, "_started", False):

        camera.start()



    try:

        if not have_external_frame:

            cv_image, depth_frame, color_intrinsics = camera.capture()

            if cv_image is None:

                print("[Spine] Failed to capture frame from RealSense.", flush=True)

                return None



        img, detections = detect_spines(image=cv_image, conf_threshold=conf_threshold, refine_sam=False)

        print(f"[Spine] {len(detections)} spine candidate(s) detected.", flush=True)



        best = _pick_best_match(img, detections, target_title)



        center_xyz, pick_xyz = None, None

        if best is not None:

            if refine_sam:

                best = _refine_target_with_sam(img, best)



            u, v = best["center"]

            pick_u, pick_v = best["top_edge_center"]

            if depth_frame is not None and color_intrinsics is not None:

                from input.depth_utils import extract_robust_book_depth

                book_depth = extract_robust_book_depth(

                    depth_frame,

                    corners=best.get("corners"),

                    center=(u, v),

                    mask_contour=best.get("mask_contour")

                )

                center_xyz = deproject_target(

                    depth_frame, color_intrinsics, u, v,

                    reference_depth=book_depth,

                    corners=best.get("corners")

                )

                pick_xyz = deproject_target(

                    depth_frame, color_intrinsics, pick_u, pick_v,

                    reference_depth=book_depth,

                    corners=best.get("corners")

                )



                # Ensure pick_xyz never captures background wall (>0.65m)

                if (pick_xyz is None or pick_xyz[2] > 0.65) and book_depth is not None:

                    print(f"[Spine] Pick depth was background (>0.65m) or missing. Using robust book depth ({book_depth:.3f}m).", flush=True)

                    pick_xyz = deproject_target(

                        depth_frame, color_intrinsics, pick_u, pick_v,

                        reference_depth=book_depth

                    )



            if pick_xyz is None and center_xyz is not None:

                pick_xyz = center_xyz

            if center_xyz is None and pick_xyz is not None:

                center_xyz = pick_xyz



            best["xyz_center_camera_frame"] = center_xyz

            best["xyz_pick_camera_frame"] = pick_xyz

            best_xyz = pick_xyz if pick_xyz is not None else center_xyz

            best["xyz_camera_frame"] = best_xyz



            if save_visual:

                best["visual_frame"] = _save_visual_result(img.copy(), best, output_path, xyz=best_xyz)



        if best is None:

            return None



        target_output = {

            "u": u,

            "v": v,

            "width_px": best["width_px"],

            "top_edge_center_u": best["top_edge_center"][0],

            "top_edge_center_v": best["top_edge_center"][1],

            "pick_point_2d": best.get("pick_point_2d", best["top_edge_center"]),

            "pick_edge_endpoints": best.get("pick_edge_endpoints"),

            "pick_edge_angle_deg": best.get("pick_edge_angle_deg"),

            "grasp_approach_angle_deg": best.get("grasp_approach_angle_deg"),

            "is_short_edge": best.get("is_short_edge"),

            "pick_edge_length_px": best.get("pick_edge_length_px"),

            "similarity": best["similarity"],

            "conf": best["conf"],

            "text": best["text"],

            "xyz_camera_frame": best_xyz,

            "xyz_pick_camera_frame": pick_xyz,

            "xyz_center_camera_frame": center_xyz,

            "corners": best.get("corners"),

            "mask_contour": best.get("mask_contour"),

            "visual_frame": best.get("visual_frame"),

        }

        from shared.display_utils import print_target_summary

        print_target_summary(target_output)



        return target_output



    except Exception as e:

        print(f"\n[Spine] RealSense Camera error: {e}", flush=True)

        return None



    finally:

        if owns_camera and camera is not None:

            camera.stop()





if __name__ == "__main__":

    print("=== Step 1: Reader Input Query ===", flush=True)

    query_result = get_book_query()



    if query_result:

        target_title = query_result["title"]

        print(f"\n=== Step 2: Finding Target Spine for '{target_title}' ===", flush=True)

        find_target_spine_live(target_title)

    else:

        print("\n[Spine] No valid query received.")

