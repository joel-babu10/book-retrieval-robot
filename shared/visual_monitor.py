"""
shared/visual_monitor.py
----------------------------------------------------------------------------
Real-Time Visual Monitor and Diagnostics Dashboard for Book Retrieval Robot.
Provides a native OpenCV desktop HUD window with:
  - Live RealSense RGB camera feed with bounding boxes, SAM mask, and pick crosshair
  - Glassmorphic telemetry cards (Camera optical XYZ, Dobot base XYZ, grasp angle, reach)
  - Live 8-stage Dobot retrieval progress bar and stage status updates
  - Instant high-visibility error diagnostics alerts (unreachable targets, depth holes, OCR misses)
----------------------------------------------------------------------------
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _draw_translucent_rect(img, x, y, w, h, bg_color=(20, 22, 28), alpha=0.75, border_color=(70, 75, 85), thickness=1):
    """Draws a smooth, translucent rounded rectangle card."""
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w_img - 1, int(x + w)), min(h_img - 1, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return

    sub_img = img[y1:y2, x1:x2]
    color_rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    blended = cv2.addWeighted(color_rect, alpha, sub_img, 1.0 - alpha, 0)
    img[y1:y2, x1:x2] = blended

    if border_color and thickness > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)


class VisualMonitor:
    def __init__(self, window_name="Book Retrieval Robot - Autonomous Monitor", enabled=True):
        self.window_name = window_name
        self.enabled = enabled
        self.window_created = False
        self.current_frame = None
        self.base_annotated_frame = None
        self.query_title = ""
        self.mode = "auto"
        self.target = None
        self.robot_coords = None      # [x, y, z, r] mm, deg
        self.reach_ok = True
        self.reach_reason = "ok"
        self.stage_num = 0            # 1 to 8
        self.status_msg = "Initializing..."
        self.is_error = False
        self.error_detail = ""
        self.is_success = False
        self.last_rendered_frame = None

    def show_startup(self, status_msg="Listening for book query (speak anytime or enter title)..."):
        """Displays initial startup dashboard before camera frame is acquired."""
        self.status_msg = status_msg
        canvas = np.full((720, 1280, 3), (16, 20, 26), dtype=np.uint8)
        for y in range(0, 720, 40):
            cv2.line(canvas, (0, y), (1280, y), (24, 28, 38), 1)
        _draw_translucent_rect(canvas, 320, 220, 640, 240, bg_color=(20, 26, 38), alpha=0.92, border_color=(0, 240, 255), thickness=2)
        cv2.putText(canvas, "AUTONOMOUS BOOK RETRIEVAL ROBOT", (350, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "SYSTEM READY & LISTENING", (490, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, status_msg, (350, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "RealSense D435i | Dobot Magician | RapidOCR | SAM Vit-B", (350, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 155, 175), 1, cv2.LINE_AA)

        self.current_frame = canvas
        self.base_annotated_frame = canvas.copy()
        if self.enabled:
            self._ensure_window()
        self.render()

    def show_frame(self, raw_frame, query_title="", status="Camera frame acquired. Processing...", mode="auto"):
        """Displays the newly captured live camera frame immediately in the HUD."""
        if query_title:
            self.query_title = query_title
        self.mode = mode
        self.status_msg = status
        if raw_frame is not None:
            h, w = raw_frame.shape[:2]
            if w > 1440 or h > 900:
                scale = min(1440 / w, 900 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                self.current_frame = cv2.resize(raw_frame.copy(), (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.scale_x = float(new_w) / float(w)
                self.scale_y = float(new_h) / float(h)
            else:
                self.current_frame = raw_frame.copy()
                self.scale_x = 1.0
                self.scale_y = 1.0
            self.base_annotated_frame = self.current_frame.copy()
        if self.enabled:
            self._ensure_window()
        self.render()

    def setup_target(self, raw_frame, target, query_title, mode="spine", robot_coords=None):
        """Initializes the monitor with camera frame, target metadata, and kinematics."""
        self.query_title = query_title
        self.mode = mode
        self.target = target
        self.robot_coords = robot_coords

        if raw_frame is None:
            # Create a blank fallback canvas
            raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Work on a copy sized nicely (e.g. 1280x720 or native)
        h, w = raw_frame.shape[:2]
        if w > 1440 or h > 900:
            scale = min(1440 / w, 900 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            self.current_frame = cv2.resize(raw_frame.copy(), (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.scale_x = float(new_w) / float(w)
            self.scale_y = float(new_h) / float(h)
        else:
            self.current_frame = raw_frame.copy()
            self.scale_x = 1.0
            self.scale_y = 1.0

        # Check kinematics if robot_coords available
        if robot_coords is not None:
            rx, ry, rz = robot_coords[:3]
            try:
                import dobot_kin as K
                self.reach_ok, self.reach_reason, _ = K.check(rx, ry, rz)
            except Exception:
                r_dist = float(np.hypot(rx, ry))
                self.reach_ok = (140.0 <= r_dist <= 325.0 and -50.0 <= rz <= 170.0)
                self.reach_reason = "ok" if self.reach_ok else f"r={r_dist:.1f}mm outside [140, 325]"

        self._render_scene_annotations()
        if self.enabled:
            self._ensure_window()
        self.render()

    def _ensure_window(self):
        if not self.enabled or self.window_created:
            return
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Size window comfortably on screen
            cv2.resizeWindow(self.window_name, 1280, 720)
            self.window_created = True
        except Exception as e:
            print(f"[VisualMonitor] Note: could not open GUI window: {e}", flush=True)
            self.enabled = False

    def _render_scene_annotations(self):
        """Renders bounding box, SAM mask, pick crosshair, and edge vector onto base frame."""
        if self.current_frame is None:
            return

        img = self.current_frame
        h_img, w_img = img.shape[:2]
        target = self.target

        if target is None:
            self.base_annotated_frame = img.copy()
            return

        sx = getattr(self, "scale_x", 1.0)
        sy = getattr(self, "scale_y", 1.0)

        # 1. Translucent SAM Mask Shading
        sam_cnt = target.get("mask_contour")
        if sam_cnt is not None and len(sam_cnt) >= 3:
            try:
                cnt_pts = np.asarray(sam_cnt, dtype=np.float32).copy()
                if cnt_pts.ndim == 2:
                    cnt_pts = cnt_pts.reshape((-1, 1, 2))
                cnt_pts[:, :, 0] *= sx
                cnt_pts[:, :, 1] *= sy
                cnt_pts = cnt_pts.astype(np.int32)
                mask_overlay = img.copy()
                cv2.fillPoly(mask_overlay, [cnt_pts], (0, 242, 254))  # Glowing cyan fill
                img = cv2.addWeighted(mask_overlay, 0.25, img, 0.75, 0)
                cv2.drawContours(img, [cnt_pts], -1, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        # 2. Draw Bounding Box (OBB) in vibrant neon green
        corners = target.get("corners")
        if corners is None:
            corners = target.get("obb_corners")
        if corners is not None:
            try:
                pts = np.asarray(corners, dtype=np.float32).copy()
                pts[:, 0] *= sx
                pts[:, 1] *= sy
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            except Exception:
                pass

        # 3. Pick Edge & Endpoints
        endpoints = target.get("pick_edge_endpoints")
        if endpoints and len(endpoints) == 2:
            p1 = (int(endpoints[0][0] * sx), int(endpoints[0][1] * sy))
            p2 = (int(endpoints[1][0] * sx), int(endpoints[1][1] * sy))
            cv2.line(img, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(img, p1, 5, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p2, 5, (255, 255, 0), -1, cv2.LINE_AA)

        # 4. Target Pick Point (Glowing Crosshair)
        pick_u = target.get("top_edge_center_u") or target.get("u")
        pick_v = target.get("top_edge_center_v") or target.get("v")
        if pick_u is not None and pick_v is not None:
            pu, pv = int(pick_u * sx), int(pick_v * sy)
            # Glowing double-circle target
            cv2.circle(img, (pu, pv), 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (pu, pv), 5, (0, 0, 255), -1, cv2.LINE_AA)
            # White crosshair
            cv2.drawMarker(img, (pu, pv), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2, line_type=cv2.LINE_AA)

            # Draw grasp alignment direction indicator line (angle R)
            grasp_deg = target.get("pick_edge_angle_deg")
            if grasp_deg is not None:
                ang_rad = np.radians(grasp_deg)
                length = 35.0
                dx = int(length * np.cos(ang_rad))
                dy = int(length * np.sin(ang_rad))
                cv2.line(img, (pu - dx, pv - dy), (pu + dx, pv + dy), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (pu + dx, pv + dy), 3, (0, 255, 255), -1)

            # Pick label badge (show true full-resolution coordinates in text label)
            orig_pu = int(pick_u)
            orig_pv = int(pick_v)
            badge_text = f"PICK POINT ({orig_pu}, {orig_pv})"
            cv2.putText(img, badge_text, (pu + 18, max(25, pv - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, badge_text, (pu + 18, max(25, pv - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Center marker
        cu, cv = target.get("u"), target.get("v")
        if cu is not None and cv is not None:
            cv2.drawMarker(img, (int(cu * sx), int(cv * sy)), (255, 120, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2, line_type=cv2.LINE_AA)

        self.base_annotated_frame = img.copy()

    def render(self):
        """Composes complete HUD onto the frame and displays in the OpenCV window."""
        if self.current_frame is None:
            return None

        frame = (self.base_annotated_frame if self.base_annotated_frame is not None else self.current_frame).copy()
        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. TOP HEADER BAR
        # -------------------------------------------------------------
        bar_h = 42
        _draw_translucent_rect(frame, 0, 0, w, bar_h, bg_color=(12, 14, 20), alpha=0.90, border_color=(40, 50, 65), thickness=1)

        # Brand / Title
        cv2.putText(frame, "AUTONOMOUS BOOK RETRIEVAL ROBOT", (15, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 1, cv2.LINE_AA)

        # Mode Badge
        mode_str = f"MODE: {self.mode.upper()}"
        cv2.putText(frame, mode_str, (430, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # System Hardware Status
        hw_status = "CUDA: RTX 4060 | CAMERA: RealSense D435i | DOBOT: COM7"
        (tw, _), _ = cv2.getTextSize(hw_status, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, hw_status, (w - tw - 20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 190, 210), 1, cv2.LINE_AA)

        # Glowing green hardware dot
        cv2.circle(frame, (w - tw - 32, 22), 5, (0, 255, 0), -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. LEFT TELEMETRY CARD - VISION & OCR MATCH
        # -------------------------------------------------------------
        card_w, card_h = 340, 185
        card_x, card_y = 15, 52
        _draw_translucent_rect(frame, card_x, card_y, card_w, card_h, bg_color=(16, 20, 28), alpha=0.82, border_color=(50, 70, 95), thickness=1)

        cv2.putText(frame, "TARGET DETECTION & RECOGNITION", (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (card_x + 12, card_y + 28), (card_x + card_w - 12, card_y + 28), (50, 70, 95), 1)

        query_show = self.query_title if self.query_title else "N/A"
        if len(query_show) > 26:
            query_show = query_show[:24] + "..."
        cv2.putText(frame, f"Query Target:  \"{query_show}\"", (card_x + 12, card_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        if self.target:
            matched_ocr = self.target.get("text", "Unknown")
            if len(matched_ocr) > 26:
                matched_ocr = matched_ocr[:24] + "..."
            sim = self.target.get("similarity", 0.0)
            conf = self.target.get("conf", 0.0)
            bw = self.target.get("width_px", 0.0)
            sam_done = "YES (Sub-pixel)" if self.target.get("mask_contour") is not None else "Standard"

            sim_color = (0, 255, 0) if sim >= 0.50 else ((0, 220, 255) if sim >= 0.35 else (0, 0, 255))

            cv2.putText(frame, f"OCR Matched:   \"{matched_ocr}\"", (card_x + 12, card_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Match Score:   {sim:.2f}", (card_x + 12, card_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sim_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Detector Conf: {conf:.2f}", (card_x + 12, card_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Spine Width:   {bw:.1f} px", (card_x + 12, card_y + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"SAM Refined:   {sam_done}", (card_x + 12, card_y + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STATUS: No target matched yet", (card_x + 12, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 150, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. RIGHT TELEMETRY CARD - 3D SPATIAL & ROBOT KINEMATICS
        # -------------------------------------------------------------
        rcard_w, rcard_h = 380, 185
        rcard_x = w - rcard_w - 15
        rcard_y = 52
        _draw_translucent_rect(frame, rcard_x, rcard_y, rcard_w, rcard_h, bg_color=(16, 20, 28), alpha=0.82, border_color=(50, 70, 95), thickness=1)

        cv2.putText(frame, "3D SPATIAL & ROBOT KINEMATICS", (rcard_x + 12, rcard_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (rcard_x + 12, rcard_y + 28), (rcard_x + rcard_w - 12, rcard_y + 28), (50, 70, 95), 1)

        cam_xyz = None
        if self.target:
            cam_xyz = self.target.get("xyz_pick_camera_frame")
            if cam_xyz is None:
                cam_xyz = self.target.get("xyz_center_camera_frame")

        if cam_xyz is not None:
            cx, cy, cz = cam_xyz
            cv2.putText(frame, f"Camera Frame:  ({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) m", (rcard_x + 12, rcard_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Camera Frame:  Awaiting Depth...", (rcard_x + 12, rcard_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1, cv2.LINE_AA)

        if self.robot_coords is not None:
            rx, ry, rz, rr = self.robot_coords
            r_reach = float(np.hypot(rx, ry))
            reach_col = (0, 255, 0) if self.reach_ok else (0, 0, 255)
            reach_badge = "REACHABLE [140 - 325 mm]" if self.reach_ok else f"UNREACHABLE ({self.reach_reason})"
            if len(reach_badge) > 34:
                reach_badge = reach_badge[:32] + "..."

            j1_deg = float(np.degrees(np.arctan2(ry, rx)))
            cv2.putText(frame, f"Dobot Target:  X={rx:.1f}, Y={ry:.1f}, Z={rz:.1f} mm", (rcard_x + 12, rcard_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Grasp & Base:  J1={j1_deg:.1f} deg | R={rr:.1f} deg", (rcard_x + 12, rcard_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Reach Radius:  r = {r_reach:.1f} mm", (rcard_x + 12, rcard_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Workspace:     {reach_badge}", (rcard_x + 12, rcard_y + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.44, reach_col, 1, cv2.LINE_AA)
            cv2.putText(frame, f"J2 Clearance:  Max Height Z=165.0 mm", (rcard_x + 12, rcard_y + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Dobot Target:  Calculating...", (rcard_x + 12, rcard_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. BOTTOM INTERACTIVE STATUS & DIAGNOSTICS BANNER
        # -------------------------------------------------------------
        banner_h = 76
        banner_y = h - banner_h - 10
        banner_x = 15
        banner_w = w - 30

        # Colors based on state
        if self.is_error:
            bg_col = (15, 15, 140)       # Dark Alert Red
            border_col = (40, 40, 255)   # Vivid Red
            hdr_text = "[SYSTEM ERROR OCCURRED]"
            hdr_col = (80, 80, 255)
        elif self.is_success:
            bg_col = (15, 90, 25)        # Dark Emerald Green
            border_col = (40, 220, 60)   # Vivid Green
            hdr_text = "[SUCCESS] SEQUENCE COMPLETED"
            hdr_col = (100, 255, 120)
        else:
            bg_col = (16, 22, 35)        # Sleek Navy/Cyan
            border_col = (0, 200, 240)   # Cyan border
            if self.stage_num > 0:
                hdr_text = f"DOBOT RETRIEVAL MOTION -- STAGE [{self.stage_num}/8]"
            else:
                hdr_text = "AUTONOMOUS SYSTEM STATUS"
            hdr_col = (0, 240, 255)

        _draw_translucent_rect(frame, banner_x, banner_y, banner_w, banner_h, bg_color=bg_col, alpha=0.92, border_color=border_col, thickness=2)

        # Header of banner
        cv2.putText(frame, hdr_text, (banner_x + 16, banner_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hdr_col, 1, cv2.LINE_AA)

        # Main message
        msg = self.status_msg
        if len(msg) > 95:
            msg = msg[:92] + "..."
        cv2.putText(frame, msg, (banner_x + 16, banner_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Error detail or Stepper Pills
        if self.is_error:
            err_sub = self.error_detail or "Execution paused to preserve robot. Inspect error above. Press any key or ESC to close."
            cv2.putText(frame, err_sub, (banner_x + 16, banner_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)
        elif self.is_success:
            cv2.putText(frame, "Book successfully transferred to deposit zone. Press any key or ESC to close.", (banner_x + 16, banner_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 255, 220), 1, cv2.LINE_AA)
        else:
            # 8-stage progress tracker pills
            stages = ["1:Elevate", "2:PreRotate", "3:Transit", "4:Descend", "5:Clamp", "6:Extract", "7:Deposit", "8:Home"]
            step_x = banner_x + 16
            for i, st_name in enumerate(stages, 1):
                st_color = (0, 255, 0) if i < self.stage_num else ((0, 240, 255) if i == self.stage_num else (120, 130, 140))
                st_fmt = f"[{st_name}]" if i == self.stage_num else st_name
                cv2.putText(frame, st_fmt, (step_x, banner_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, st_color, 1, cv2.LINE_AA)
                step_x += 105

        self.last_rendered_frame = frame.copy()

        if self.enabled:
            try:
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
            except Exception:
                pass

        return frame

    def update_status(self, message: str, stage_num: int = None, is_error: bool = False, error_detail: str = "", is_success: bool = False):
        """Updates the status message and refreshes the display window immediately."""
        self.status_msg = message
        if stage_num is not None:
            self.stage_num = stage_num
        self.is_error = is_error
        if error_detail:
            self.error_detail = error_detail
        self.is_success = is_success
        self.render()

    def show_error(self, message: str, error_detail: str = ""):
        """Flags an error and displays it prominently in red."""
        self.update_status(message, is_error=True, error_detail=error_detail)
        # Pump GUI events so the user sees the error immediately
        for _ in range(5):
            cv2.waitKey(20)

    def wait_for_user(self, timeout_ms: int = 0):
        """Waits for user keypress (e.g. at end of execution or on error) before closing."""
        if not self.enabled or not self.window_created:
            return
        try:
            cv2.waitKey(timeout_ms)
        except Exception:
            pass

    def close(self):
        """Destroys the OpenCV monitor window."""
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.window_created = False


# Global singleton monitor
_global_monitor = None


def get_visual_monitor(enabled=True):
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = VisualMonitor(enabled=enabled)
    return _global_monitor
