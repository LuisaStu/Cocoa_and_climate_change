from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relative_suitability_index(
    reference_response: ArrayLike,
    period_response: ArrayLike,
    *,
    base_index: float = 100.0,
) -> NDArray[np.float64]:
    """Normalize model responses to a continuous, cellwise reference index.

    Inputs are the model's internal response quantities, not observed farm yields.
    Callers should pass matching spatial cells and temporal summaries. Invalid or
    non-positive reference values raise instead of silently creating an index.
    """
    reference, period = np.broadcast_arrays(
        np.asarray(reference_response, dtype=np.float64),
        np.asarray(period_response, dtype=np.float64),
    )
    if not np.isfinite(base_index) or base_index <= 0.0:
        raise ValueError("base_index must be finite and positive")
    if not np.isfinite(reference).all() or np.any(reference <= 0.0):
        raise ValueError("reference responses must be finite and strictly positive")
    if not np.isfinite(period).all() or np.any(period < 0.0):
        raise ValueError("period responses must be finite and non-negative")
    return np.asarray(base_index * period / reference, dtype=np.float64)
