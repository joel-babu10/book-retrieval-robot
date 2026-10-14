"""
shared package
----------------------------------------------------------------------------
Shared utilities across vision pipelines.
"""

from .sam_refine import refine_obb_with_sam, preload_sam_refine
from .ocr_reader import get_shared_ocr_reader, preload_shared_ocr_reader

__all__ = [
    "refine_obb_with_sam",
    "preload_sam_refine",
    "get_shared_ocr_reader",
    "preload_shared_ocr_reader",
]
