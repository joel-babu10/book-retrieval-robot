"""
scattered_pipeline package
----------------------------------------------------------------------------
Scattered book detection (SAM 3 text-prompted segmentation), OCR text matching,
and RealSense 3D target localization.
"""

from .find_target_scattered import find_target_scattered_live, find_target_scattered_image
from .detect_scattered import detect_books, detect_scattered, preload_scattered_model
from .ocr_match_scattered import (
    ocr_best_match,
    ocr_scattered_crop,
    compute_title_similarity,
    title_similarity,
    preload_scattered_reader,
)

__all__ = [
    "find_target_scattered_live",
    "find_target_scattered_image",
    "detect_books",
    "detect_scattered",
    "preload_scattered_model",
    "ocr_best_match",
    "ocr_scattered_crop",
    "compute_title_similarity",
    "title_similarity",
    "preload_scattered_reader",
]
