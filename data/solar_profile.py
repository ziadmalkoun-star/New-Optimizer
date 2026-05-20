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
from utils.time_utils import build_quarter_hour_index

def build_standard_france_solar_profile() -> np.ndarray:
    idx = build_quarter_hour_index(DEFAULT_YEAR)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    seasonal = 0.18 + 0.82 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    daylight_hours = 8.0 + 8.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    sunrise = 12.0 - daylight_hours / 2.0
    sunset = 12.0 + daylight_hours / 2.0
    shape = np.zeros(QH_PER_YEAR, dtype=float)

    for i in range(QH_PER_YEAR):
        if sunrise[i] <= hour[i] <= sunset[i]:
            x = (hour[i] - sunrise[i]) / max(sunset[i] - sunrise[i], 1e-9)
            shape[i] = (np.sin(np.pi * x) ** 1.6) * seasonal[i]

    total = shape.sum()
    if total <= 0:
        raise ValueError("Impossible de générer une courbe solaire standard valide.")
    return shape / total

def build_pv_generation_mwh(
    solar_profile_relative: np.ndarray,
    pv_dc_mw: float,
    productible_kwh_per_kwp: float,
    pv_losses_pct: float,
    plant_availability_pct: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    relative = _validate_array_length(solar_profile_relative, "Le profil solaire")
    relative = np.maximum(relative, 0.0)

    if relative.sum() <= 0:
        raise ValueError("Le profil solaire doit avoir une somme strictement positive.")
    if pv_dc_mw < 0 or productible_kwh_per_kwp < 0:
        raise ValueError("La puissance PV et le productible doivent être positifs.")
    if not (0 <= pv_losses_pct <= 100):
        raise ValueError("Les pertes PV doivent être entre 0 et 100 %.")
    if not (0 <= plant_availability_pct <= 100):
        raise ValueError("La disponibilité doit être entre 0 et 100 %.")

    annual_dc_mwh = pv_dc_mw * productible_kwh_per_kwp
    net_factor = (1.0 - pv_losses_pct / 100.0) * (plant_availability_pct / 100.0)
    annual_net_mwh = annual_dc_mwh * net_factor
    relative = relative / relative.sum()
    qh_net_mwh = annual_net_mwh * relative

    stats = {
        "annual_dc_mwh": float(annual_dc_mwh),
        "annual_net_mwh": float(annual_net_mwh),
        "annual_losses_mwh": float(max(annual_dc_mwh - annual_net_mwh, 0.0)),
    }
    return qh_net_mwh, stats
