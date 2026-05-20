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

def apply_tso_dso_curtailment(
    pv_hourly_mwh: np.ndarray,
    monthly_curtailment_pct: np.ndarray,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV horaire avant TSO/DSO")
    monthly_curtailment_pct = np.asarray(monthly_curtailment_pct, dtype=float).reshape(-1)

    if len(monthly_curtailment_pct) != 12:
        raise ValueError("La courbe mensuelle de curtailment TSO/DSO doit avoir 12 valeurs.")

    idx = build_quarter_hour_index(DEFAULT_YEAR)
    month_idx = idx.month.to_numpy() - 1
    pct_hourly = monthly_curtailment_pct[month_idx]

    pv_after = pv_hourly_mwh * (1.0 - pct_hourly)
    pv_after = np.maximum(pv_after, 0.0)
    curtailed = np.maximum(pv_hourly_mwh - pv_after, 0.0)
    flag = curtailed > 1e-12

    return {
        "pv_after_tso_dso_mwh": pv_after,
        "tso_dso_curtailed_mwh": curtailed,
        "tso_dso_curtailment_flag": flag.astype(int),
        "tso_dso_monthly_pct_hourly": pct_hourly,
    }

def apply_self_curtailment(
    pv_hourly_mwh: np.ndarray,
    pv_spot_price_raw: np.ndarray,
    pv_spot_price_effective: np.ndarray,
    enable_self_curtailment: bool,
    pv_commercial_structure: str,
    curtailment_threshold_eur_per_mwh: float,
    cfd_price_eur_per_mwh: float,
    negative_price_rule: bool,
    consecutive_negative_hours_limit: int,
    ppa_price_eur_per_mwh: float,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV avant self curtailment")
    pv_spot_price_raw = _validate_array_length(pv_spot_price_raw, "Prix spot PV raw")
    pv_spot_price_effective = _validate_array_length(pv_spot_price_effective, "Prix spot PV effectif")

    sellable = pv_hourly_mwh.copy()
    pv_effective_price = pv_spot_price_effective.copy()
    self_curtailed = np.zeros(QH_PER_YEAR, dtype=float)
    self_flag = np.zeros(QH_PER_YEAR, dtype=int)
    structure_arr = np.full(QH_PER_YEAR, pv_commercial_structure, dtype=object)
    reason_arr = np.full(QH_PER_YEAR, "", dtype=object)

    if not enable_self_curtailment:
        return {
            "pv_after_self_curtailment_mwh": sellable,
            "self_curtailed_mwh": self_curtailed,
            "self_curtailment_flag": self_flag,
            "pv_effective_price_eur_per_mwh": pv_effective_price,
            "pv_commercial_structure_hourly": structure_arr,
            "self_curtailment_reason": reason_arr,
        }

    if pv_commercial_structure == "Fully merchant":
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "Merchant threshold curtailment"
        pv_effective_price = pv_spot_price_effective

    elif pv_commercial_structure == "With CfD":
        pv_effective_price[:] = float(cfd_price_eur_per_mwh)

        if negative_price_rule:
            neg_run = 0
            for t in range(QH_PER_YEAR):
                if pv_spot_price_raw[t] < 0:
                    neg_run += 1
                    if neg_run > int(consecutive_negative_hours_limit):
                        self_curtailed[t] = sellable[t]
                        sellable[t] = 0.0
                        self_flag[t] = 1
                        reason_arr[t] = "CfD negative-hours curtailment"
                else:
                    neg_run = 0

    elif pv_commercial_structure == "With PPA":
        pv_effective_price[:] = float(ppa_price_eur_per_mwh)
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "PPA threshold curtailment"

    else:
        raise ValueError(f"Structure commerciale PV non reconnue: {pv_commercial_structure}")

    return {
        "pv_after_self_curtailment_mwh": sellable,
        "self_curtailed_mwh": self_curtailed,
        "self_curtailment_flag": self_flag,
        "pv_effective_price_eur_per_mwh": pv_effective_price,
        "pv_commercial_structure_hourly": structure_arr,
        "self_curtailment_reason": reason_arr,
    }
