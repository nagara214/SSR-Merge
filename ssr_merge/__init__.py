"""SSR-Merge: training-free LoRA merging via Subspace Signal Routing.

Public API
----------
- :func:`run`: run SSR on a pre-built diffsynth-style pipeline and
  return (or save) the merged LoRA.
"""

from __future__ import annotations

from .runners import run

__all__ = ["run", "__version__"]
__version__ = "0.1.0"
