from __future__ import annotations

import numpy as np


def error_to_signal_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
    error_energy = float(np.sum(np.square(reference - estimate)))
    signal_energy = float(np.sum(np.square(reference)))
    if signal_energy == 0.0:
        raise ValueError("reference signal energy is zero")
    return error_energy / signal_energy
