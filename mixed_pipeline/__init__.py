"""
mixed_pipeline
----------------------------------------------------------------------------
Dual-mode scanner: runs both spine and scattered pipelines on the same
camera frame, compares candidate match similarity, and returns the winner.
"""

from mixed_pipeline.find_target_mixed import (
    find_target_mixed_live,
    find_target_mixed_image,
)

__all__ = [
    "find_target_mixed_live",
    "find_target_mixed_image",
]
