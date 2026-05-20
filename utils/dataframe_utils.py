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

def _make_qh_dataframe(data: dict, expected_len: int = QH_PER_YEAR) -> pd.DataFrame:
    """Build a DataFrame from quarter-hour arrays, expanding scalar/1-row values.

    Streamlit output exports mix true 35,040-step arrays with scalar summary
    arrays such as np.array([annual_cap]). Pandas requires equal lengths; this
    helper expands scalars and 1-element arrays to expected_len and pads/truncates
    other mismatched arrays defensively so exports do not fail.
    """
    normalized = {}
    for key, value in data.items():
        if isinstance(value, pd.Series):
            arr = value.to_numpy()
        elif isinstance(value, pd.Index):
            arr = value.to_numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                normalized[key] = np.full(expected_len, value, dtype=object)
                continue

        if arr.ndim == 0:
            normalized[key] = np.full(expected_len, arr.item())
            continue

        arr = arr.reshape(-1)
        if len(arr) == expected_len:
            normalized[key] = arr
        elif len(arr) == 1:
            normalized[key] = np.full(expected_len, arr[0])
        elif len(arr) > expected_len:
            normalized[key] = arr[:expected_len]
        else:
            pad_value = np.nan if arr.dtype.kind in "fiu" else None
            padded = np.empty(expected_len, dtype=arr.dtype if arr.dtype.kind not in "OUS" else object)
            padded[:len(arr)] = arr
            padded[len(arr):] = pad_value
            normalized[key] = padded
    return pd.DataFrame(normalized)

def monthly_dataframe(
    result: Dict[str, np.ndarray],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    idx = build_quarter_hour_index(DEFAULT_YEAR)

    df = pd.DataFrame({
        "datetime": idx,
        "pv_direct_revenue": result["pv_direct_revenue"],
        "batt_sale_revenue": result["batt_sale_revenue"],
        "grid_charge_cost": result["grid_charge_cost"],
        "wholesale_cycle_cost": result["wholesale_cycle_cost_eur"] if "wholesale_cycle_cost_eur" in result else np.zeros(QH_PER_YEAR),
        "pv_direct_mwh": result["pv_direct"],
        "shifted_mwh": result["discharge"],
        "grid_charge_mwh": result["grid_charge"],
        "pv_to_batt_mwh": result["pv_to_batt"],
        "pv_curtailed_to_battery_mwh_actual": curtailment_outputs["pv_curtailed_to_battery_mwh_actual"],
        "pv_curtailment_candidate_mwh": curtailment_outputs["pv_curtailment_candidate_mwh"],
        "pv_curtailed_residual_lost_mwh": curtailment_outputs["pv_curtailed_residual_lost_mwh"],
        "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
        "pv_only_revenue": pure_pv_benchmark["pv_only_revenue_eur"],
        "afrr_charge_mwh": result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in result else np.zeros(QH_PER_YEAR),
        "afrr_discharge_mwh": result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in result else np.zeros(QH_PER_YEAR),
        "afrr_charge_cost": result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_sale_revenue": result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_cycle_cost": result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_net_revenue": result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_up_revenue": result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_down_revenue": result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_total_revenue": result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_up_awarded_hours": result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_down_awarded_hours": result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in result else np.zeros(QH_PER_YEAR),
        "bess_revenue_loss_due_to_capture_rate": result["bess_revenue_loss_due_to_capture_rate_hourly_eur"] if "bess_revenue_loss_due_to_capture_rate_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "bess_theoretical_revenue_without_capture": result["bess_theoretical_revenue_without_capture_hourly_eur"] if "bess_theoretical_revenue_without_capture_hourly_eur" in result else np.zeros(QH_PER_YEAR),
    })

    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    monthly = df.groupby("month", as_index=False).sum(numeric_only=True)

    monthly["bess_net_revenue"] = (
        monthly["batt_sale_revenue"]
        - monthly["grid_charge_cost"]
        + monthly["afrr_net_revenue"]
        + monthly["afrr_capacity_total_revenue"]
    )
    monthly["net_revenue"] = monthly["pv_direct_revenue"] + monthly["bess_net_revenue"]

    monthly["pv_revenue_keur_per_mw"] = monthly["pv_direct_revenue"] / max(pv_dc_mw, 1e-12) / 1000.0
    monthly["bess_revenue_keur_per_mw"] = monthly["bess_net_revenue"] / max(batt_power_mw, 1e-12) / 1000.0

    monthly["pv_revenue_eur_per_mwh"] = monthly["pv_direct_revenue"] / monthly["pv_direct_mwh"].clip(lower=1e-12)
    monthly["bess_total_discharged_mwh"] = monthly["shifted_mwh"] + monthly["afrr_discharge_mwh"]
    monthly["bess_revenue_eur_per_mwh"] = monthly["bess_net_revenue"] / monthly["bess_total_discharged_mwh"].clip(lower=1e-12)

    return monthly

def build_inputs_dataframe(inputs: SimulationInputs) -> pd.DataFrame:
    rows = [
        ("batt_power_mw", inputs.batt_power_mw),
        ("bess_usable_power_mw", inputs.batt_power_mw),
        ("bess_duration_h", inputs.bess_duration_h),
        ("bess_usable_capacity_mwh", inputs.nominal_batt_energy_mwh),
        ("bess_gross_capacity_mwh", inputs.gross_batt_energy_mwh),
        ("technical_eta_charge", inputs.technical_eta_charge),
        ("technical_eta_discharge", inputs.technical_eta_discharge),
        ("bess_availability_pct", inputs.bess_availability_pct),
        ("effective_batt_energy_mwh", inputs.batt_energy_mwh),
        ("pv_dc_mw", inputs.pv_dc_mw),
        ("productible_kwh_per_kwp", inputs.productible_kwh_per_kwp),
        ("pv_losses_pct", inputs.pv_losses_pct),
        ("plant_availability_pct", inputs.plant_availability_pct),
        ("dispatch_eta_charge", inputs.eta_charge),
        ("dispatch_eta_discharge", inputs.eta_discharge),
        ("nightly_bess_revenue_eur", inputs.nightly_bess_revenue_eur),
        ("soc_steps", inputs.soc_steps),
        ("initial_soc_mwh", inputs.initial_soc_mwh),
        ("final_soc_mwh", inputs.final_soc_mwh),
        ("min_soc_pct", inputs.min_soc_pct),
        ("max_soc_pct", inputs.max_soc_pct),
        ("grid_export_limit_mw", inputs.grid_export_limit_mw),
        ("cycle_cost_eur_per_mwh", inputs.cycle_cost_eur_per_mwh),
        ("charge_quantile", inputs.charge_quantile),
        ("discharge_quantile", inputs.discharge_quantile),
        ("max_cycles_per_year", inputs.max_cycles_per_year),
        ("min_spread_arbitrage_eur_per_mwh", inputs.min_spread_arbitrage_eur_per_mwh),
        ("forward_optimization_horizon_hours", inputs.forward_optimization_horizon_hours),
        ("afrr_up_cross_market_min_spread_eur_per_mwh", inputs.afrr_up_cross_market_min_spread_eur_per_mwh),
        ("afrr_down_to_wholesale_min_spread_eur_per_mwh", inputs.afrr_down_to_wholesale_min_spread_eur_per_mwh),
        ("pv_capture_rate_pct", inputs.pv_capture_rate_pct),
        ("bess_capture_rate_pct", inputs.bess_capture_rate_pct),
        ("enable_afrr", inputs.enable_afrr),
        ("afrr_min_spread_eur_per_mwh", inputs.afrr_min_spread_eur_per_mwh),
        ("afrr_cycle_cost_eur_per_mwh", inputs.afrr_cycle_cost_eur_per_mwh),
        ("afrr_max_events_per_day", inputs.afrr_max_events_per_day),
        ("afrr_energy_start_hour", inputs.afrr_night_start_hour),
        ("afrr_energy_end_hour", inputs.afrr_night_end_hour),
        ("afrr_pv_zero_tolerance_mwh", inputs.afrr_pv_zero_tolerance_mwh),
        ("afrr_n_qh_per_side", inputs.afrr_n_qh_per_side),
        ("afrr_energy_down_activation_pct", inputs.afrr_energy_down_activation_pct),
        ("afrr_energy_up_activation_pct", inputs.afrr_energy_up_activation_pct),
        ("enable_afrr_capacity", inputs.enable_afrr_capacity),
        ("afrr_certified_capacity_pct", inputs.afrr_certified_capacity_pct),
        ("afrr_capacity_success_rate_pct", inputs.afrr_capacity_success_rate_pct),
        ("allow_afrr_energy_without_capacity", inputs.allow_afrr_energy_without_capacity),
        ("afrr_certified_capacity_up_mw", inputs.afrr_certified_capacity_up_mw),
        ("afrr_certified_capacity_down_mw", inputs.afrr_certified_capacity_down_mw),
        ("afrr_capacity_start_hour", inputs.afrr_capacity_start_hour),
        ("afrr_capacity_end_hour", inputs.afrr_capacity_end_hour),
        ("enable_tso_dso_curtailment", inputs.enable_tso_dso_curtailment),
        ("enable_self_curtailment", inputs.enable_self_curtailment),
        ("curtailment_threshold_eur_per_mwh", inputs.curtailment_threshold_eur_per_mwh),
        ("pv_commercial_structure", inputs.pv_commercial_structure),
        ("cfd_price_eur_per_mwh", inputs.cfd_price_eur_per_mwh),
        ("negative_price_rule", inputs.negative_price_rule),
        ("consecutive_negative_hours_limit", inputs.consecutive_negative_hours_limit),
        ("ppa_price_eur_per_mwh", inputs.ppa_price_eur_per_mwh),
        ("charge_battery_if_curtailment", inputs.charge_battery_if_curtailment),
        ("enable_cfd", inputs.enable_cfd),
        ("cfd_price_standalone_eur_per_mwh", inputs.cfd_price_standalone_eur_per_mwh),
        ("enable_ppa", inputs.enable_ppa),
        ("ppa_price_standalone_eur_per_mwh", inputs.ppa_price_standalone_eur_per_mwh),
        ("bess_degradation_curve_pct", "" if inputs.bess_degradation_curve_pct is None else list(inputs.bess_degradation_curve_pct)),
        ("degraded_bess_energy_by_year_mwh", "" if inputs.degraded_bess_energy_by_year_mwh is None else list(inputs.degraded_bess_energy_by_year_mwh)),
        ("project_lifetime_years", inputs.project_lifetime_years),
        ("grid_import_fee_eur_per_mwh", inputs.grid_import_fee_eur_per_mwh),
        ("grid_export_fee_eur_per_mwh", inputs.grid_export_fee_eur_per_mwh),
        ("omie_buy_fee_eur_per_mwh", inputs.omie_buy_fee_eur_per_mwh),
        ("omie_sell_fee_eur_per_mwh", inputs.omie_sell_fee_eur_per_mwh),
        ("ree_system_fee_eur_per_mwh", inputs.ree_system_fee_eur_per_mwh),
        ("imbalance_cost_pv_eur_per_mwh", inputs.imbalance_cost_pv_eur_per_mwh),
        ("imbalance_cost_bess_eur_per_mwh", inputs.imbalance_cost_bess_eur_per_mwh),
        ("afrr_capacity_fee_pct", inputs.afrr_capacity_fee_pct),
        ("afrr_energy_fee_pct", inputs.afrr_energy_fee_pct),
        ("afrr_energy_fee_eur_per_mwh", inputs.afrr_energy_fee_eur_per_mwh),
        ("ivpee_generation_tax_pct", inputs.ivpee_generation_tax_pct),
        ("apply_ivpee_to_pv", inputs.apply_ivpee_to_pv),
        ("apply_ivpee_to_bess_export", inputs.apply_ivpee_to_bess_export),
        ("apply_ivpee_to_afrr_energy", inputs.apply_ivpee_to_afrr_energy),
        ("apply_ivpee_to_afrr_capacity", inputs.apply_ivpee_to_afrr_capacity),
        ("corporate_tax_pct", inputs.corporate_tax_pct),
        ("withholding_tax_pct", inputs.withholding_tax_pct),
        ("local_fixed_tax_eur_per_year", inputs.local_fixed_tax_eur_per_year),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])
