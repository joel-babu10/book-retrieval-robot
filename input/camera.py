"""
input/camera.py
----------------------------------------------------------------------------
RealSense D435i camera handling — start/stop pipeline, capture aligned
high-resolution color+depth frame pairs, expose intrinsics.

Optimized for high-quality OCR text reading:
  - Default Color Stream: 1920x1080 (Full HD @ 30 FPS)
  - Default Depth Stream: 1280x720 (30 FPS)
  - Auto-Exposure & White-Balance stabilization (30 warmup frames)
"""

import numpy as np
import pyrealsense2 as rs

DEFAULT_COLOR_WIDTH = 1920
DEFAULT_COLOR_HEIGHT = 1080
DEFAULT_DEPTH_WIDTH = 1280
DEFAULT_DEPTH_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_WARMUP_FRAMES = 30  # Allow auto-exposure & white balance to settle fully


class RealSenseCamera:
    def __init__(self, color_width=DEFAULT_COLOR_WIDTH, color_height=DEFAULT_COLOR_HEIGHT,
                 depth_width=DEFAULT_DEPTH_WIDTH, depth_height=DEFAULT_DEPTH_HEIGHT,
                 fps=DEFAULT_FPS, warmup_frames=DEFAULT_WARMUP_FRAMES):
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.warmup_frames = warmup_frames

        self.pipeline = None
        self.align = None
        self._started = False

    def start(self, verbose=False):
        if self._started:
            return

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)

            print(f"[Camera] Starting RealSense pipeline ({self.color_width}x{self.color_height} color)...", flush=True)
            profile = self.pipeline.start(config)

            # Enable auto-exposure and auto-white-balance on color sensor
            sensors = profile.get_device().query_sensors()
            for s in sensors:
                if s.is_color_sensor():
                    if s.supports(rs.option.enable_auto_exposure):
                        s.set_option(rs.option.enable_auto_exposure, 1)
                    if s.supports(rs.option.enable_auto_white_balance):
                        s.set_option(rs.option.enable_auto_white_balance, 1)

            self.align = rs.align(rs.stream.color)
            self._started = True

            # Warmup frames to let auto-exposure and white balance stabilize
            for _ in range(self.warmup_frames):
                self.pipeline.wait_for_frames()

        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start RealSense camera: {e}")

    def stop(self):
        if self._started and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self._started = False
            print("[Camera] RealSense pipeline stopped.", flush=True)

    def capture(self):
        """
        Grabs one aligned color+depth frame pair.

        Returns:
            cv_image: np.ndarray (BGR, 1080x1920x3) — high-res for detection & OCR
            depth_frame: pyrealsense2 depth frame — for deprojection
            color_intrinsics: pyrealsense2 intrinsics — for deprojection
            or (None, None, None) if a frame pair could not be grabbed.
        """
        if not self._started:
            raise RuntimeError("Camera not started — call start() first, or use RealSenseCamera() as a context manager.")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("[Camera] Failed to grab aligned color/depth frames.", flush=True)
            return None, None, None

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        cv_image = np.asanyarray(color_frame.get_data())

        return cv_image, depth_frame, color_intrinsics

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
