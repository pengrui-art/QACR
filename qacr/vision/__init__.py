"""Vision backbone components for QACR."""

from .multipath_depth import DepthMultiPathExecutor
from .highres_reencode import HighResReEncoder

__all__ = ["DepthMultiPathExecutor", "HighResReEncoder"]
