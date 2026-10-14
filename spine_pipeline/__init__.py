"""
spine_pipeline package
----------------------------------------------------------------------------
Spine detection, OCR text matching, and RealSense 3D target localization.
"""

from .find_target_spine import find_target_spine_live, find_target_spine_image
from .detect_spine import detect_spines, preload_spine_model
from .ocr_match_spine import ocr_spine_crop, title_similarity

__all__ = [
    "find_target_spine_live",
    "find_target_spine_image",
    "detect_spines",
    "preload_spine_model",
    "ocr_spine_crop",
    "title_similarity",
]
