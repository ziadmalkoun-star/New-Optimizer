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
from data.validators import _validate_array_length
from utils.math_utils import rolling_forward_max

def compute_forward_cross_market_value_curves(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    """Build practical forward-looking value curves for cross-market charging.

    This is intentionally lightweight and compatible with the existing DP.
    It does not replace the full dispatch optimizer with LP/MILP; it provides
    forward opportunity signals used as charging gates and audit columns.
    """
    horizon_steps = int(max(1, round(float(inputs.forward_optimization_horizon_hours) * QH_PER_HOUR)))

    wholesale_value = _validate_array_length(inputs.batt_sell_price, "BESS sell price", QH_PER_YEAR) * (float(inputs.bess_capture_rate_pct) / 100.0)

    if inputs.enable_afrr and inputs.afrr_discharge_price_qh is not None:
        afrr_up_energy = _validate_array_length(inputs.afrr_discharge_price_qh, "aFRR UP energy price", QH_PER_YEAR)
    else:
        afrr_up_energy = np.full(QH_PER_YEAR, -1e30, dtype=float)

    if inputs.enable_afrr_capacity and inputs.afrr_capacity_up_price_h is not None:
        cap_up = _validate_array_length(inputs.afrr_capacity_up_price_h, "aFRR UP capacity price", QH_PER_YEAR)
        success = min(max(float(inputs.afrr_capacity_success_rate_pct) / 100.0, 0.0), 1.0)
        activation_up = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
        if activation_up > 1e-9:
            # Convert capacity availability value into an expected EUR/MWh uplift
            # on activated UP energy. This is an expected-value signal only.
            cap_uplift_per_mwh = cap_up * success / activation_up
        else:
            cap_uplift_per_mwh = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_up_value = afrr_up_energy + cap_uplift_per_mwh
    else:
        afrr_up_value = afrr_up_energy.copy()

    future_wholesale = rolling_forward_max(wholesale_value, horizon_steps)
    future_afrr_up = rolling_forward_max(afrr_up_value, horizon_steps)
    future_best = np.maximum(future_wholesale, future_afrr_up)
    future_type = np.where(future_afrr_up > future_wholesale, "afrr_up", "wholesale")
    future_type[future_best <= -1e20] = "none"

    return {
        "future_expected_wholesale_value_eur_per_mwh": future_wholesale,
        "future_expected_afrr_up_value_eur_per_mwh": future_afrr_up,
        "future_best_market_value_eur_per_mwh": future_best,
        "future_best_market_type": future_type.astype(object),
        "forward_horizon_hours": np.full(QH_PER_YEAR, float(inputs.forward_optimization_horizon_hours), dtype=float),
    }
