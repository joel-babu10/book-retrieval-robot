"""
input package
----------------------------------------------------------------------------
Reader query input, RealSense camera interface, and 3D depth deprojection.
"""

from .query_input import get_book_query, match_query
from .camera import RealSenseCamera
from .depth_utils import deproject_target, median_depth_patch

__all__ = [
    "get_book_query",
    "match_query",
    "RealSenseCamera",
    "deproject_target",
    "median_depth_patch",
]
