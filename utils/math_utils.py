from typing import Dict, Tuple
from collections import deque
import io
import time
from dataclasses import replace
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

from config import *
from models.simulation_inputs import SimulationInputs

def rolling_forward_max(values: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Maximum future value within (t, t + horizon_steps] in O(n).

    This replaces the previous O(n * horizon) loop with a monotonic deque.
    It produces the same forward-looking maximum used by the optimizer, but
    is much faster when the forward horizon is large.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    out = np.full(n, -1e30, dtype=float)
    h = int(max(1, horizon_steps))
    cleaned = np.nan_to_num(values, nan=-1e30, posinf=1e30, neginf=-1e30)

    dq: deque[int] = deque()
    for t in range(n - 1, -1, -1):
        # Remove indices outside the future window (t, t+h].
        max_idx = t + h
        while dq and dq[0] > max_idx:
            dq.popleft()

        # Add t+1 to the candidate window.
        add_idx = t + 1
        if add_idx < n:
            add_val = cleaned[add_idx]
            while dq and cleaned[dq[-1]] <= add_val:
                dq.pop()
            dq.append(add_idx)

        if dq:
            out[t] = float(cleaned[dq[0]])

    return out

def _make_flat_curve(value: float, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    if value is None:
        raise ValueError("La valeur moyenne annuelle n'a pas été renseignée.")
    return np.full(expected_len, float(value), dtype=float)
