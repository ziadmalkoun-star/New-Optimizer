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

def build_pure_pv_benchmark(
    pv_generation_mwh: np.ndarray,
    pv_price: np.ndarray,
    grid_export_limit_mw: float,
) -> Dict[str, np.ndarray]:
    pv_generation_mwh = _validate_array_length(pv_generation_mwh, "Production PV benchmark")
    pv_price = _validate_array_length(pv_price, "Prix PV benchmark")

    pv_only_direct_mwh = np.minimum(np.maximum(pv_generation_mwh, 0.0), float(grid_export_limit_mw) * QH_DT_HOURS)
    pv_only_revenue_eur = pv_only_direct_mwh * pv_price
    total_pv_only_revenue_eur = float(pv_only_revenue_eur.sum())

    return {
        "pv_only_direct_mwh": pv_only_direct_mwh,
        "pv_only_revenue_eur": pv_only_revenue_eur,
        "total_pv_only_revenue_eur": np.array([total_pv_only_revenue_eur]),
    }
