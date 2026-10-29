"""
mixed_pipeline/find_target_mixed.py
----------------------------------------------------------------------------
Dual-Arrangement Mixed Mode:
Runs BOTH the spine pipeline (shelf OBB) and the scattered pipeline (desk OBB)
on the exact same captured camera frame, compares query similarity scores,
and picks the higher-confidence winning target.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spine_pipeline.find_target_spine import (
    find_target_spine_live,
    find_target_spine_image,
)
from scattered_pipeline.find_target_scattered import (
    find_target_scattered_live,
    find_target_scattered_image,
)


def _render_mixed_visual_result(cv_image, winner_target, winner_mode, spine_score, scattered_score, output_path):
    """
    Renders high-visibility bounding box and comparison overlay for mixed mode.
    """
    img = cv_image.copy()
    img_h, img_w = img.shape[:2]

    # Retrieve or reconstruct corners
    corners = winner_target.get("corners")
    if corners is None:
        corners = winner_target.get("obb_corners")

    # Fallback reconstruction if explicit corners were not provided
    if corners is None and "u" in winner_target and "width_px" in winner_target and "height_px" in winner_target:
        cx, cy = float(winner_target["u"]), float(winner_target["v"])
        w, h = float(winner_target["width_px"]), float(winner_target["height_px"])
        angle = float(winner_target.get("obb_angle", 0.0))
        rect = ((cx, cy), (w, h), angle)
        corners = cv2.boxPoints(rect)

    # 1. Translucent SAM Mask Shading & Organic Contour
    sam_cnt = winner_target.get("mask_contour")
    if sam_cnt is not None and len(sam_cnt) >= 3:
        mask_overlay = img.copy()
        cv2.fillPoly(mask_overlay, [sam_cnt.astype(np.int32)], (0, 242, 254))  # Glowing cyan fill
        img = cv2.addWeighted(mask_overlay, 0.22, img, 0.78, 0)
        cv2.drawContours(img, [sam_cnt.astype(np.int32)], -1, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Geometric Target Bounding Box
    if corners is not None:
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        box_color = (0, 0, 255) if winner_mode == "SPINE" else (0, 255, 0)
        cv2.polylines(img, [pts], isClosed=True, color=box_color, thickness=4)

    cx, cy = winner_target["u"], winner_target["v"]
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    # Highlight robot pick edge (grasp edge) and pick point
    top_u = winner_target.get("top_edge_center_u", cx)
    top_v = winner_target.get("top_edge_center_v", cy)
    endpoints = winner_target.get("pick_edge_endpoints")
    if endpoints and len(endpoints) == 2:
        p1 = (int(endpoints[0][0]), int(endpoints[0][1]))
        p2 = (int(endpoints[1][0]), int(endpoints[1][1]))
        cv2.line(img, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)  # Cyan pick edge
        cv2.circle(img, p1, 5, (255, 255, 0), -1)
        cv2.circle(img, p2, 5, (255, 255, 0), -1)

    # Robot pick point marker (red circle with white cross)
    cv2.circle(img, (int(top_u), int(top_v)), 7, (0, 0, 255), -1)
    cv2.drawMarker(img, (int(top_u), int(top_v)), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.putText(img, "PICK", (int(top_u) - 15, int(top_v) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    info_lines = [
        f"MIXED WINNER: {winner_mode}",
        f"OCR: {winner_target.get('text', '')[:28]}",
        f"Match Score: {winner_target.get('similarity', 0.0):.2f}",
        f"Spine vs Scat: {spine_score:.2f} vs {scattered_score:.2f}",
        f"Center (u,v): ({cx:.1f}, {cy:.1f})",
        f"Pick Edge (u,v): ({top_u:.1f}, {top_v:.1f})",
    ]
    pick_angle = winner_target.get("pick_edge_angle_deg")
    if pick_angle is not None:
        info_lines.append(f"Grasp Angle: {pick_angle:.1f} deg")

    pick_xyz = winner_target.get("xyz_pick_camera_frame")
    if pick_xyz is None:
        pick_xyz = winner_target.get("xyz_camera_frame")
    if pick_xyz is not None:
        info_lines.append(f"Pick XYZ: ({pick_xyz[0]:.3f}, {pick_xyz[1]:.3f}, {pick_xyz[2]:.3f}) m")

    box_w, box_h = 280, 22 * len(info_lines) + 12
    text_x = int(cx) + 15
    text_y = int(cy) - 60
    text_x = max(5, min(text_x, img_w - box_w - 5))
    text_y = max(30, min(text_y, img_h - box_h - 5))

    overlay = img.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - 20), (text_x + box_w, text_y + box_h - 15), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    for i, line in enumerate(info_lines):
        color = (0, 255, 255) if i == 0 else (0, 255, 0)
        cv2.putText(img, line, (text_x, text_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"\nSaved visual result to: {output_path}", flush=True)

    try:
        from datetime import datetime
        runs_dir = Path(output_path).parent / "saved_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(output_path).stem
        history_path = runs_dir / f"{stem}_{timestamp}.jpg"
        cv2.imwrite(str(history_path), img)
        print(f"Stored image copy to: {history_path}", flush=True)
    except Exception as e:
        print(f"Warning storing image history: {e}", flush=True)


def _is_flat_desk_target(t):
    """Checks if a target detection represents a book lying flat on the desk surface."""
    if not t:
        return False
    corners = t.get("corners")
    if corners is None:
        corners = t.get("obb_corners")
    if corners is not None and len(corners) == 4:
        c = np.asarray(corners)
        dx = float(np.ptp(c[:, 0]))
        dy = float(np.ptp(c[:, 1]))
        w = float(min(t.get("width_px", dx), t.get("height_px", dy)))
        h = float(max(t.get("width_px", dx), t.get("height_px", dy)))
        aspect = h / max(w, 1.0)
        cy = float(t.get("center", (0, 0))[1])
        y_top = float(np.min(c[:, 1]))

        # Horizontal book across desk
        if dx >= 1.15 * dy:
            return True
        # Wide cover face visible
        if w >= 170.0 and aspect <= 2.2:
            return True
        # Desk surface location without tall shelf height
        if (cy > 750.0 or y_top > 550.0) and dy < 350.0:
            return True
    return False


def _is_upright_shelf_target(t):
    """Checks if a target detection is an upright book/spine on a shelf."""
    if not t:
        return False
    corners = t.get("corners")
    if corners is None:
        corners = t.get("obb_corners")
    if corners is not None and len(corners) == 4:
        c = np.asarray(corners)
        dx = float(np.ptp(c[:, 0]))
        dy = float(np.ptp(c[:, 1]))
        w = float(min(t.get("width_px", dx), t.get("height_px", dy)))
        h = float(max(t.get("width_px", dx), t.get("height_px", dy)))
        aspect = h / max(w, 1.0)
        y_top = float(np.min(c[:, 1]))
        cy = float(t.get("center", (0, 0))[1])

        # Tall vertical column reaching shelf height
        if dy >= 350.0 and y_top < 550.0 and dy >= 1.8 * dx and aspect >= 2.0 and cy < 820.0:
            return True
    return False


def _evaluate_winner(spine_target, scattered_target, query_title):
    """
    Compares spine and scattered detection results and selects the winning candidate.
    Uses geometric classification to properly classify flat desk books vs upright shelf spines.
    """
    spine_score = spine_target.get("similarity", 0.0) if spine_target else 0.0
    scattered_score = scattered_target.get("similarity", 0.0) if scattered_target else 0.0

    print("\n---- [Mixed Mode Comparison] ----", flush=True)
    if spine_target:
        s_text = spine_target.get('text', '')[:35]
        s_angle = spine_target.get('pick_edge_angle_deg') or spine_target.get('obb_angle') or 0.0
        print(f"  [Spine]     match={spine_score:.2f}  conf={spine_target.get('conf', 0.0):.2f}  angle={float(s_angle):.1f}°  OCR='{s_text}'", flush=True)
    else:
        print("  [Spine]     No candidate matched above threshold.", flush=True)

    if scattered_target:
        sc_text = scattered_target.get('text', '')[:35]
        sc_angle = scattered_target.get('pick_edge_angle_deg') or scattered_target.get('obb_angle') or 0.0
        print(f"  [Scattered] match={scattered_score:.2f}  conf={scattered_target.get('conf', 0.0):.2f}  angle={float(sc_angle):.1f}°  OCR='{sc_text}'", flush=True)
    else:
        print("  [Scattered] No candidate matched above threshold.", flush=True)

    if spine_target is None and scattered_target is None:
        print("\n[Mixed Mode] Neither pipeline found a book matching query above threshold.", flush=True)
        return None, None, 0.0, 0.0

    if spine_target is not None and scattered_target is None:
        winner_target = spine_target
        winner_mode = "SCATTERED" if _is_flat_desk_target(spine_target) else "SPINE"
    elif scattered_target is not None and spine_target is None:
        winner_target = scattered_target
        winner_mode = "SPINE" if _is_upright_shelf_target(scattered_target) else "SCATTERED"
    else:
        # Both matched: check if they detect the SAME physical book
        sp_cx = spine_target.get("u", spine_target.get("center", (0, 0))[0])
        sp_cy = spine_target.get("v", spine_target.get("center", (0, 0))[1])
        sc_cx = scattered_target.get("u", scattered_target.get("center", (0, 0))[0])
        sc_cy = scattered_target.get("v", scattered_target.get("center", (0, 0))[1])
        dist = np.hypot(sp_cx - sc_cx, sp_cy - sc_cy)

        # Check if spine center is enclosed within scattered polygon
        sc_corners = scattered_target.get("corners")
        if sc_corners is None:
            sc_corners = scattered_target.get("obb_corners")
        spine_inside_scattered = False
        if sc_corners is not None:
            dist_poly = cv2.pointPolygonTest(np.array(sc_corners, dtype=np.float32), (float(sp_cx), float(sp_cy)), False)
            if dist_poly >= 0:
                spine_inside_scattered = True

        same_book = (dist < 180 or spine_inside_scattered)

        if same_book:
            # When both models detected the same book, determine physical arrangement:
            # If it's a flat book lying on the desk, SCATTERED pipeline provides the accurate full outline
            if _is_flat_desk_target(scattered_target) or not _is_upright_shelf_target(spine_target):
                print(f"\n[Mixed Disambiguation] Book at (u={sc_cx:.1f}, v={sc_cy:.1f}) is a flat surface book -> SCATTERED mode.", flush=True)
                winner_mode = "SCATTERED"
                winner_target = scattered_target
            else:
                print(f"\n[Mixed Disambiguation] Book at (u={sp_cx:.1f}, v={sp_cy:.1f}) is an upright shelf spine -> SPINE mode.", flush=True)
                winner_mode = "SPINE"
                winner_target = spine_target
        else:
            # Different books: compare query similarity scores
            if scattered_score > (spine_score + 0.05):
                winner_target = scattered_target
                winner_mode = "SPINE" if _is_upright_shelf_target(scattered_target) else "SCATTERED"
            elif spine_score > (scattered_score + 0.05):
                winner_target = spine_target
                winner_mode = "SCATTERED" if _is_flat_desk_target(spine_target) else "SPINE"
            else:
                # Tied scores: check detection confidence
                if spine_target.get("conf", 0.0) >= scattered_target.get("conf", 0.0):
                    winner_target = spine_target
                    winner_mode = "SCATTERED" if _is_flat_desk_target(spine_target) else "SPINE"
                else:
                    winner_target = scattered_target
                    winner_mode = "SPINE" if _is_upright_shelf_target(scattered_target) else "SCATTERED"

    winner_target["arrangement_mode"] = winner_mode.lower()
    winner_target["pipeline_winner"] = winner_mode
    print(f"\n[Mixed Mode Winner] >>> {winner_mode} <<< (Score: {winner_target.get('similarity', 0.0):.2f})", flush=True)
    return winner_target, winner_mode, spine_score, scattered_score


def _is_confirmed_vertical_shelf_spine(target):
    """Checks if a target is unambiguously a vertical shelf spine."""
    if not target:
        return False
    score = target.get("similarity", 0.0)
    if score < 0.85:
        return False
    return _is_upright_shelf_target(target)


def find_target_mixed_live(query_title, camera=None, cv_image=None, depth_frame=None,
                            color_intrinsics=None, refine_sam=True, save_visual=True,
                            output_path="result_target_mixed_live.jpg"):
    """
    Captures ONE aligned camera frame, runs both the Spine pipeline and Scattered
    pipeline on pristine copies, and returns the highest scoring target.
    Includes early exit when a confirmed vertical shelf spine is matched.
    """
    from input.camera import RealSenseCamera

    have_external_frame = (cv_image is not None and depth_frame is not None and color_intrinsics is not None)
    owns_camera = (camera is None and not have_external_frame)

    if owns_camera:
        camera = RealSenseCamera()
        camera.start()
    elif camera is not None and not getattr(camera, "_started", False):
        camera.start()

    try:
        if not have_external_frame:
            print("[Mixed] Capturing single aligned RGB + Depth frame...", flush=True)
            cv_image, depth_frame, color_intrinsics = camera.capture()
            if cv_image is None:
                print("[Mixed] Failed to capture frame from RealSense.", flush=True)
                return None

        # 1. Run Spine Pipeline on isolated copy (candidate search without SAM)
        print("\n--- Running Spine Pipeline in Mixed Mode ---", flush=True)
        spine_img = cv_image.copy()
        spine_target = find_target_spine_live(
            query_title,
            camera=camera,
            cv_image=spine_img,
            depth_frame=depth_frame,
            color_intrinsics=color_intrinsics,
            save_visual=False,
            refine_sam=False,
        )

        scattered_target = None
        # Prioritized Early Exit: If an unambiguous shelf spine matched with >= 0.88, skip scattered!
        if _is_confirmed_vertical_shelf_spine(spine_target):
            print(f"\n[Mixed Early Exit] Confirmed vertical shelf spine matched with high confidence ({spine_target['similarity']:.2f}). Skipping scattered scan!", flush=True)
        else:
            # 2. Run Scattered Pipeline on isolated copy
            print("\n--- Running Scattered Pipeline in Mixed Mode ---", flush=True)
            scattered_img = cv_image.copy()
            scattered_target = find_target_scattered_live(
                query_title,
                camera=camera,
                cv_image=scattered_img,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                save_visual=False,
                refine_sam=False,
            )

        # 3. Compare candidates and select winner
        winner_target, winner_mode, spine_score, scattered_score = _evaluate_winner(
            spine_target, scattered_target, query_title
        )

        if winner_target is None:
            return None

        # 4. Run SAM Refinement ONCE exclusively on the single winning target
        if refine_sam and "corners" in winner_target and winner_target["corners"] is not None:
            from shared.sam_refine import refine_obb_with_sam
            print(f"\n[SAM Refine] Target matched ({winner_mode}). Running fast ROI SAM refinement on winner...", flush=True)
            res = refine_obb_with_sam(cv_image, winner_target["corners"], return_contour=True)
            if isinstance(res, tuple):
                refined_corners, sam_contour = res
            else:
                refined_corners, sam_contour = res, None

            if refined_corners is not None:
                winner_target["corners"] = refined_corners
                winner_target["obb_corners"] = refined_corners
                if sam_contour is not None:
                    winner_target["mask_contour"] = sam_contour

                from shared.sam_refine import extract_refined_pick_edge
                pick_loc = "spine_mid" if winner_mode == "SCATTERED" else "top_left"
                edge_info = extract_refined_pick_edge(refined_corners, pick_location=pick_loc, img=cv_image, query_title=query_title)
                pick_u, pick_v = edge_info["pick_point_2d"]
                endpoints = edge_info["pick_edge_endpoints"]
                edge_len = edge_info["pick_edge_length_px"]

                winner_target["pick_point_2d"] = (pick_u, pick_v)
                winner_target["top_edge_center_u"] = pick_u
                winner_target["top_edge_center_v"] = pick_v
                winner_target["pick_edge_endpoints"] = endpoints
                winner_target["pick_edge_angle_deg"] = edge_info["pick_edge_angle_deg"]
                winner_target["grasp_approach_angle_deg"] = edge_info["grasp_approach_angle_deg"]
                winner_target["is_short_edge"] = edge_info["is_short_edge"]
                winner_target["pick_edge_length_px"] = edge_len
                winner_target["obb_angle"] = edge_info["pick_edge_angle_deg"]

                center_u, center_v = edge_info["center_2d"]
                winner_target["u"] = center_u
                winner_target["v"] = center_v

                # Re-deproject depth with refined center and pick edge
                if depth_frame is not None and color_intrinsics is not None:
                    from input.depth_utils import deproject_target, extract_robust_book_depth
                    book_depth = extract_robust_book_depth(
                        depth_frame,
                        corners=winner_target.get("corners"),
                        center=(center_u, center_v),
                        mask_contour=winner_target.get("mask_contour")
                    )
                    center_xyz = deproject_target(
                        depth_frame, color_intrinsics, center_u, center_v,
                        reference_depth=book_depth,
                        corners=winner_target.get("corners")
                    )
                    pick_xyz = deproject_target(
                        depth_frame, color_intrinsics, pick_u, pick_v,
                        reference_depth=book_depth,
                        corners=winner_target.get("corners")
                    )
                    if (pick_xyz is None or pick_xyz[2] > 0.46) and book_depth is not None:
                        print(f"[Mixed] Pick depth was background (>0.46m) or missing. Using robust book depth ({book_depth:.3f}m).", flush=True)
                        pick_xyz = deproject_target(
                            depth_frame, color_intrinsics, pick_u, pick_v,
                            reference_depth=book_depth
                        )
                    winner_target["xyz_center_camera_frame"] = center_xyz
                    winner_target["xyz_pick_camera_frame"] = pick_xyz
                    winner_target["xyz_camera_frame"] = pick_xyz or center_xyz
                    if center_xyz is not None:
                        winner_target["X"] = center_xyz[0]
                        winner_target["Y"] = center_xyz[1]
                        winner_target["Z"] = center_xyz[2]

        # 5. Save unified visual result (only winner is rendered)
        if save_visual:
            _render_mixed_visual_result(
                cv_image, winner_target, winner_mode, spine_score, scattered_score, output_path
            )

        from shared.display_utils import print_target_summary
        print_target_summary(winner_target)
        return winner_target

    finally:
        if owns_camera and camera is not None:
            camera.stop()


def find_target_mixed_image(image_path, query_title, refine_sam=True, save_visual=True,
                             output_path="result_target_mixed.jpg"):
    """
    Runs dual spine + scattered scan on a saved image file.
    Fast candidate scan without SAM, then runs SAM once on the winning target.
    """
    img_path = Path(image_path)
    cv_image = cv2.imread(str(img_path))
    if cv_image is None:
        print(f"[Mixed] Could not read image: {image_path}", flush=True)
        return None

    print(f"\n[Mixed] Running dual scan on image: {img_path.name} for '{query_title}'", flush=True)

    spine_target = find_target_spine_image(
        str(img_path), query_title, save_visual=False, refine_sam=False
    )
    scattered_target = None
    if _is_confirmed_vertical_shelf_spine(spine_target):
        print(f"\n[Mixed Early Exit] Confirmed vertical shelf spine matched ({spine_target['similarity']:.2f}). Skipping scattered scan!", flush=True)
    else:
        scattered_target = find_target_scattered_image(
            str(img_path), query_title, save_visual=False, refine_sam=False
        )

    winner_target, winner_mode, spine_score, scattered_score = _evaluate_winner(
        spine_target, scattered_target, query_title
    )
    if winner_target is None:
        return None

    # Run SAM Refinement ONCE exclusively on the winning target
    if refine_sam and "corners" in winner_target and winner_target["corners"] is not None:
        from shared.sam_refine import refine_obb_with_sam
        print(f"\n[SAM Refine] Target matched ({winner_mode}). Running fast ROI SAM refinement on winner...", flush=True)
        res = refine_obb_with_sam(cv_image, winner_target["corners"], return_contour=True)
        if isinstance(res, tuple):
            refined_corners, sam_contour = res
        else:
            refined_corners, sam_contour = res, None

        if refined_corners is not None:
            winner_target["corners"] = refined_corners
            winner_target["obb_corners"] = refined_corners
            if sam_contour is not None:
                winner_target["mask_contour"] = sam_contour

            from shared.sam_refine import extract_refined_pick_edge
            pick_loc = "spine_mid" if winner_mode == "SCATTERED" else "top_left"
            edge_info = extract_refined_pick_edge(refined_corners, pick_location=pick_loc, img=cv_image, query_title=query_title)
            pick_u, pick_v = edge_info["pick_point_2d"]
            winner_target["pick_point_2d"] = (pick_u, pick_v)
            winner_target["top_edge_center_u"] = pick_u
            winner_target["top_edge_center_v"] = pick_v
            winner_target["pick_edge_endpoints"] = edge_info["pick_edge_endpoints"]
            winner_target["pick_edge_angle_deg"] = edge_info["pick_edge_angle_deg"]
            winner_target["grasp_approach_angle_deg"] = edge_info["grasp_approach_angle_deg"]
            winner_target["is_short_edge"] = edge_info["is_short_edge"]
            winner_target["pick_edge_length_px"] = edge_info["pick_edge_length_px"]
            winner_target["obb_angle"] = edge_info["pick_edge_angle_deg"]

            center_u, center_v = edge_info["center_2d"]
            winner_target["u"] = center_u
            winner_target["v"] = center_v

    if save_visual:
        _render_mixed_visual_result(
            cv_image, winner_target, winner_mode, spine_score, scattered_score, output_path
        )

    from shared.display_utils import print_target_summary
    print_target_summary(winner_target)

    return winner_target

