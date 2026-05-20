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
from utils.dataframe_utils import _make_qh_dataframe

def build_combined_soc_with_afrr(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray] | None,
    batt_energy_mwh: float,
    initial_soc_mwh: float,
    eta_charge: float,
    eta_discharge: float,
    min_soc_pct: float = 0.0,
    max_soc_pct: float = 100.0,
) -> Dict[str, np.ndarray]:

    # The wholesale dispatch is already calculated at 15-minute resolution.
    # Values are MWh per 15-minute step, so do NOT repeat or divide them again.
    wholesale_pv_to_batt_qh = _validate_array_length(
        result_hourly["pv_to_batt"], "PV vers batterie wholesale 15 min", QH_PER_YEAR
    )
    wholesale_pv_curtailed_to_batt_qh = _validate_array_length(
        result_hourly.get("pv_curtailed_to_battery", np.zeros(QH_PER_YEAR)),
        "PV curtailed vers batterie wholesale 15 min",
        QH_PER_YEAR,
    )
    wholesale_grid_charge_qh = _validate_array_length(
        result_hourly["grid_charge"], "Charge réseau wholesale 15 min", QH_PER_YEAR
    )
    wholesale_discharge_market_qh = _validate_array_length(
        result_hourly["discharge"], "Décharge wholesale 15 min", QH_PER_YEAR
    )

    if afrr_result is not None:
        afrr_charge_market_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float)
        afrr_discharge_market_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float)
    else:
        afrr_charge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_discharge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)

    # Convert to SOC flows
    wholesale_charge_to_soc_qh = (
        wholesale_pv_to_batt_qh
        + wholesale_pv_curtailed_to_batt_qh
        + wholesale_grid_charge_qh
    ) * eta_charge
    wholesale_discharge_from_soc_qh = wholesale_discharge_market_qh / max(eta_discharge, 1e-12)

    afrr_charge_to_soc_qh = afrr_charge_market_qh * eta_charge
    afrr_discharge_from_soc_qh = afrr_discharge_market_qh / max(eta_discharge, 1e-12)

    combined_charge_to_soc_qh = wholesale_charge_to_soc_qh + afrr_charge_to_soc_qh
    combined_discharge_from_soc_qh = wholesale_discharge_from_soc_qh + afrr_discharge_from_soc_qh

    # SOC simulation
    min_soc_mwh = batt_energy_mwh * min_soc_pct / 100.0
    max_soc_mwh = batt_energy_mwh * max_soc_pct / 100.0

    soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    soc_qh[0] = min(max(float(initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    for t in range(QH_PER_YEAR):
        soc_next = soc_qh[t] + combined_charge_to_soc_qh[t] - combined_discharge_from_soc_qh[t]
        soc_qh[t + 1] = min(max(soc_next, min_soc_mwh), max_soc_mwh)

    soc_hourly_end = soc_qh[4::4]

    return {
        "combined_soc_qh": soc_qh,
        "combined_soc_hourly_end": soc_hourly_end,
        "combined_charge_to_soc_qh": combined_charge_to_soc_qh,
        "combined_discharge_from_soc_qh": combined_discharge_from_soc_qh,
        "wholesale_charge_to_soc_qh": wholesale_charge_to_soc_qh,
        "wholesale_pv_curtailed_to_batt_qh": wholesale_pv_curtailed_to_batt_qh,
        "wholesale_discharge_from_soc_qh": wholesale_discharge_from_soc_qh,
        "afrr_charge_to_soc_qh": afrr_charge_to_soc_qh,
        "afrr_discharge_from_soc_qh": afrr_discharge_from_soc_qh,
        "afrr_charge_market_qh": afrr_charge_market_qh,
        "afrr_discharge_market_qh": afrr_discharge_market_qh,
    }

def reconcile_wholesale_afrr_dispatch_qh(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)

    pv_direct_qh = np.asarray(result_hourly["pv_direct"], dtype=float)
    wholesale_pv_to_batt_qh = np.asarray(result_hourly["pv_to_batt"], dtype=float)
    wholesale_pv_curtailed_to_batt_qh = np.asarray(result_hourly.get("pv_curtailed_to_battery", np.zeros(QH_PER_YEAR)), dtype=float)
    wholesale_grid_charge_qh = np.asarray(result_hourly["grid_charge"], dtype=float)
    wholesale_discharge_qh = np.asarray(result_hourly["discharge"], dtype=float)

    batt_sell_price_qh = np.asarray(inputs.batt_sell_price, dtype=float)
    grid_buy_price_qh = np.asarray(inputs.grid_buy_price, dtype=float)
    afrr_charge_price_qh = np.asarray(inputs.afrr_charge_price_qh, dtype=float)
    afrr_discharge_price_qh = np.asarray(inputs.afrr_discharge_price_qh, dtype=float)

    afrr_charge_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float).copy()
    afrr_discharge_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float).copy()
    down_activated_qh = np.asarray(afrr_result.get("afrr_energy_down_activated_qh", np.zeros(QH_PER_YEAR)), dtype=int)
    up_activated_qh = np.asarray(afrr_result.get("afrr_energy_up_activated_qh", np.zeros(QH_PER_YEAR)), dtype=int)
    up_activation_shortfall_qh = np.asarray(afrr_result.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)), dtype=float)
    down_activation_shortfall_qh = np.asarray(afrr_result.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)), dtype=float)
    afrr_energy_eligible_qh = np.asarray(afrr_result.get("afrr_energy_eligible_qh", np.ones(QH_PER_YEAR, dtype=int)), dtype=int)

    if inputs.afrr_capacity_selected_market_h is None:
        afrr_capacity_selected_market_h = np.full(QH_PER_YEAR, "none", dtype=object)
    else:
        afrr_capacity_selected_market_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
        if len(afrr_capacity_selected_market_h) != QH_PER_YEAR:
            raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")
    afrr_capacity_selected_market_qh = afrr_capacity_selected_market_h

    corrected_wholesale_pv_to_batt_qh = wholesale_pv_to_batt_qh.copy()
    corrected_wholesale_pv_curtailed_to_batt_qh = wholesale_pv_curtailed_to_batt_qh.copy()
    corrected_wholesale_grid_charge_qh = wholesale_grid_charge_qh.copy()
    corrected_wholesale_discharge_qh = wholesale_discharge_qh.copy()
    corrected_afrr_charge_qh = afrr_charge_qh.copy()
    corrected_afrr_discharge_qh = afrr_discharge_qh.copy()

    selected_charge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_charge_price_qh = np.full(QH_PER_YEAR, np.nan, dtype=float)
    selected_discharge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_discharge_price_qh = np.full(QH_PER_YEAR, np.nan, dtype=float)
    stored_energy_cost_qh = np.nan_to_num(np.asarray(result_hourly.get("avg_stored_charge_price", np.zeros(QH_PER_YEAR + 1)), dtype=float).reshape(-1)[:QH_PER_YEAR], nan=0.0, posinf=0.0, neginf=0.0)
    effective_discharge_value_qh = np.zeros(QH_PER_YEAR, dtype=float)
    spread_condition_respected_qh = np.zeros(QH_PER_YEAR, dtype=int)
    wholesale_discharge_spread_ok_qh = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_up_discharge_spread_ok_qh = np.zeros(QH_PER_YEAR, dtype=int)

    export_limit_qh_mwh = inputs.grid_export_limit_mw * QH_DT_HOURS
    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    combined_soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    combined_soc_qh[0] = min(max(float(inputs.initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    for t in range(QH_PER_YEAR):
        capacity_market = afrr_capacity_selected_market_qh[t]

        if capacity_market in ("up", "down"):
            corrected_wholesale_pv_to_batt_qh[t] = 0.0
            corrected_wholesale_pv_curtailed_to_batt_qh[t] = 0.0
            corrected_wholesale_grid_charge_qh[t] = 0.0
            corrected_wholesale_discharge_qh[t] = 0.0
            if capacity_market == "down":
                corrected_afrr_discharge_qh[t] = 0.0
            elif capacity_market == "up":
                corrected_afrr_charge_qh[t] = 0.0
        elif inputs.enable_afrr_capacity and not inputs.allow_afrr_energy_without_capacity:
            # If aFRR Energy without awarded Capacity is not allowed, remove all aFRR Energy
            # in quarter-hours where the battery did not receive an aFRR Capacity award.
            corrected_afrr_charge_qh[t] = 0.0
            corrected_afrr_discharge_qh[t] = 0.0
        else:
            # Mode 2: aFRR and wholesale compete as routes for the same physical battery.
            # This branch also allows aFRR Energy without Capacity when the checkbox is ticked.
            if corrected_afrr_charge_qh[t] > 1e-12:
                if afrr_charge_price_qh[t] < grid_buy_price_qh[t]:
                    corrected_wholesale_grid_charge_qh[t] = 0.0
                else:
                    corrected_afrr_charge_qh[t] = 0.0
            if corrected_afrr_discharge_qh[t] > 1e-12:
                if afrr_discharge_price_qh[t] > batt_sell_price_qh[t]:
                    corrected_wholesale_discharge_qh[t] = 0.0
                else:
                    corrected_afrr_discharge_qh[t] = 0.0

        export_room_qh = max(export_limit_qh_mwh - pv_direct_qh[t], 0.0)
        total_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        if total_discharge_qh > export_room_qh + 1e-12:
            scale = export_room_qh / max(total_discharge_qh, 1e-12)
            corrected_wholesale_discharge_qh[t] *= scale
            corrected_afrr_discharge_qh[t] *= scale

        total_charge_qh = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        )
        total_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]

        # Never charge and discharge simultaneously. Keep the economically stronger selected route.
        if total_charge_qh > 1e-12 and total_discharge_qh > 1e-12:
            charge_saving = max(grid_buy_price_qh[t] - afrr_charge_price_qh[t], 0.0) if corrected_afrr_charge_qh[t] > 1e-12 else 0.0
            discharge_uplift = max(afrr_discharge_price_qh[t] - batt_sell_price_qh[t], 0.0) if corrected_afrr_discharge_qh[t] > 1e-12 else 0.0
            if discharge_uplift >= charge_saving:
                corrected_wholesale_pv_to_batt_qh[t] = 0.0
                corrected_wholesale_pv_curtailed_to_batt_qh[t] = 0.0
                corrected_wholesale_grid_charge_qh[t] = 0.0
                corrected_afrr_charge_qh[t] = 0.0
            else:
                corrected_wholesale_discharge_qh[t] = 0.0
                corrected_afrr_discharge_qh[t] = 0.0

        soc_now = combined_soc_qh[t]
        total_charge_input = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        )
        max_charge_input_by_headroom = max(max_soc_mwh - soc_now, 0.0) / max(inputs.eta_charge, 1e-12)
        if total_charge_input > max_charge_input_by_headroom + 1e-12:
            scale = max_charge_input_by_headroom / max(total_charge_input, 1e-12)
            corrected_wholesale_pv_to_batt_qh[t] *= scale
            corrected_wholesale_pv_curtailed_to_batt_qh[t] *= scale
            corrected_wholesale_grid_charge_qh[t] *= scale
            corrected_afrr_charge_qh[t] *= scale

        total_discharge_output = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        max_discharge_output_by_soc = max(soc_now - min_soc_mwh, 0.0) * inputs.eta_discharge
        if total_discharge_output > max_discharge_output_by_soc + 1e-12:
            scale = max_discharge_output_by_soc / max(total_discharge_output, 1e-12)
            corrected_wholesale_discharge_qh[t] *= scale
            corrected_afrr_discharge_qh[t] *= scale

        # Enforce minimum spread before final discharge into wholesale or aFRR UP.
        cost_per_output = stored_energy_cost_qh[t] / max(inputs.eta_discharge, 1e-12)
        wholesale_spread_t = batt_sell_price_qh[t] - cost_per_output - inputs.cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        afrr_up_spread_t = afrr_discharge_price_qh[t] - cost_per_output - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        if corrected_wholesale_discharge_qh[t] > 1e-12:
            wholesale_discharge_spread_ok_qh[t] = int(wholesale_spread_t + 1e-12 >= inputs.min_spread_arbitrage_eur_per_mwh)
            if not wholesale_discharge_spread_ok_qh[t]:
                corrected_wholesale_discharge_qh[t] = 0.0
        if corrected_afrr_discharge_qh[t] > 1e-12:
            afrr_up_discharge_spread_ok_qh[t] = int(afrr_up_spread_t + 1e-12 >= inputs.afrr_min_spread_eur_per_mwh)
            if not afrr_up_discharge_spread_ok_qh[t]:
                corrected_afrr_discharge_qh[t] = 0.0
        if corrected_wholesale_discharge_qh[t] > 1e-12 or corrected_afrr_discharge_qh[t] > 1e-12:
            spread_condition_respected_qh[t] = 1
            effective_discharge_value_qh[t] = max(
                batt_sell_price_qh[t] if corrected_wholesale_discharge_qh[t] > 1e-12 else -1e30,
                afrr_discharge_price_qh[t] if corrected_afrr_discharge_qh[t] > 1e-12 else -1e30,
            )

        if corrected_afrr_charge_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "afrr"
            selected_charge_price_qh[t] = afrr_charge_price_qh[t]
        elif corrected_wholesale_grid_charge_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "wholesale_grid"
            selected_charge_price_qh[t] = grid_buy_price_qh[t]
        elif corrected_wholesale_pv_to_batt_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "pv"
        elif corrected_wholesale_pv_curtailed_to_batt_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "curtailed_pv"

        if corrected_afrr_discharge_qh[t] > 1e-12:
            selected_discharge_market_qh[t] = "afrr"
            selected_discharge_price_qh[t] = afrr_discharge_price_qh[t]
        elif corrected_wholesale_discharge_qh[t] > 1e-12:
            selected_discharge_market_qh[t] = "wholesale"
            selected_discharge_price_qh[t] = batt_sell_price_qh[t]

        charge_to_soc = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        ) * inputs.eta_charge
        discharge_from_soc = (
            corrected_wholesale_discharge_qh[t]
            + corrected_afrr_discharge_qh[t]
        ) / max(inputs.eta_discharge, 1e-12)
        combined_soc_qh[t + 1] = min(max(soc_now + charge_to_soc - discharge_from_soc, min_soc_mwh), max_soc_mwh)

    corrected_wholesale_batt_sale_revenue_qh = corrected_wholesale_discharge_qh * batt_sell_price_qh
    corrected_wholesale_grid_charge_cost_qh = corrected_wholesale_grid_charge_qh * grid_buy_price_qh
    corrected_afrr_charge_cost_qh = corrected_afrr_charge_qh * afrr_charge_price_qh
    corrected_afrr_sale_revenue_qh = corrected_afrr_discharge_qh * afrr_discharge_price_qh
    corrected_afrr_cycle_cost_qh = (corrected_afrr_discharge_qh / max(inputs.eta_discharge, 1e-12)) * inputs.afrr_cycle_cost_eur_per_mwh
    corrected_afrr_net_revenue_qh = corrected_afrr_sale_revenue_qh - corrected_afrr_charge_cost_qh  # aFRR cycle cost reference-only, not deducted

    charge_to_soc_qh = (
        corrected_wholesale_pv_to_batt_qh
        + corrected_wholesale_pv_curtailed_to_batt_qh
        + corrected_wholesale_grid_charge_qh
        + corrected_afrr_charge_qh
    ) * inputs.eta_charge
    discharge_from_soc_qh = (corrected_wholesale_discharge_qh + corrected_afrr_discharge_qh) / max(inputs.eta_discharge, 1e-12)

    def reshape_sum(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR).sum(axis=1)

    def reshape_last(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR)[:, -1]

    return {
        "datetime_qh": idx_qh,
        "wholesale_pv_to_batt_qh_mwh": corrected_wholesale_pv_to_batt_qh,
        "wholesale_pv_curtailed_to_batt_qh_mwh": corrected_wholesale_pv_curtailed_to_batt_qh,
        "wholesale_pv_curtailed_to_batt_hourly_mwh": corrected_wholesale_pv_curtailed_to_batt_qh,
        "wholesale_grid_charge_qh_mwh": corrected_wholesale_grid_charge_qh,
        "wholesale_discharge_qh_mwh": corrected_wholesale_discharge_qh,
        "wholesale_batt_sale_revenue_qh_eur": corrected_wholesale_batt_sale_revenue_qh,
        "wholesale_grid_charge_cost_qh_eur": corrected_wholesale_grid_charge_cost_qh,
        "afrr_charge_qh_mwh": corrected_afrr_charge_qh,
        "afrr_discharge_qh_mwh": corrected_afrr_discharge_qh,
        "afrr_charge_cost_qh_eur": corrected_afrr_charge_cost_qh,
        "afrr_sale_revenue_qh_eur": corrected_afrr_sale_revenue_qh,
        "afrr_cycle_cost_qh_eur": corrected_afrr_cycle_cost_qh,
        "afrr_net_revenue_qh_eur": corrected_afrr_net_revenue_qh,
        "afrr_energy_down_activated_qh": down_activated_qh,
        "afrr_energy_up_activated_qh": up_activated_qh,
        "afrr_energy_eligible_qh": afrr_energy_eligible_qh,
        "afrr_up_activation_shortfall_qh_mwh": up_activation_shortfall_qh,
        "afrr_down_activation_shortfall_qh_mwh": down_activation_shortfall_qh,
        "selected_charge_market_qh": selected_charge_market_qh,
        "selected_charge_price_qh": selected_charge_price_qh,
        "selected_discharge_market_qh": selected_discharge_market_qh,
        "selected_discharge_channel_qh": selected_discharge_market_qh,
        "selected_discharge_price_qh": selected_discharge_price_qh,
        "afrr_capacity_selected_market_qh": afrr_capacity_selected_market_qh,
        "combined_charge_to_soc_qh_mwh": charge_to_soc_qh,
        "combined_discharge_from_soc_qh_mwh": discharge_from_soc_qh,
        "combined_soc_qh": combined_soc_qh,
        "combined_soc_hourly_end_mwh": combined_soc_qh[1:],
        "wholesale_pv_to_batt_hourly_mwh": corrected_wholesale_pv_to_batt_qh,
        "wholesale_grid_charge_hourly_mwh": corrected_wholesale_grid_charge_qh,
        "wholesale_discharge_hourly_mwh": corrected_wholesale_discharge_qh,
        "wholesale_batt_sale_revenue_hourly_eur": corrected_wholesale_batt_sale_revenue_qh,
        "wholesale_grid_charge_cost_hourly_eur": corrected_wholesale_grid_charge_cost_qh,
        "afrr_charge_hourly_mwh": corrected_afrr_charge_qh,
        "afrr_discharge_hourly_mwh": corrected_afrr_discharge_qh,
        "afrr_charge_cost_hourly_eur": corrected_afrr_charge_cost_qh,
        "afrr_sale_revenue_hourly_eur": corrected_afrr_sale_revenue_qh,
        "afrr_cycle_cost_hourly_eur": corrected_afrr_cycle_cost_qh,
        "afrr_net_revenue_hourly_eur": corrected_afrr_net_revenue_qh,
        "afrr_energy_down_activated_hourly": down_activated_qh,
        "afrr_energy_up_activated_hourly": up_activated_qh,
        "stored_energy_cost_eur_per_mwh": stored_energy_cost_qh,
        "effective_discharge_value_eur_per_mwh": effective_discharge_value_qh,
        "spread_condition_respected": spread_condition_respected_qh,
        "wholesale_discharge_spread_ok": wholesale_discharge_spread_ok_qh,
        "afrr_up_discharge_spread_ok": afrr_up_discharge_spread_ok_qh,
    }

def enforce_hard_annual_cycle_cap_on_reconciliation(
    reconciliation: Dict[str, np.ndarray],
    inputs: SimulationInputs,
    afrr_capacity_result: Dict[str, np.ndarray] | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Enforce max_cycles_per_year as a hard annual discharge budget.

    The existing DP limits wholesale discharge, but final reconciliation can add
    aFRR UP discharge on top. This pass ranks all final discharge candidates
    (wholesale and aFRR UP) by net EUR/MWh value, keeps the best candidates
    within the annual discharge cap, clips the marginal interval if needed, and
    rejects the rest. It then recomputes revenues, SOC and audit columns.
    """
    if reconciliation is None:
        return reconciliation, {"wholesale_rejected": 0, "afrr_rejected": 0}

    out: Dict[str, np.ndarray] = {}
    for key, value in reconciliation.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    wh = _validate_array_length(out.get("wholesale_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "Wholesale discharge for cycle cap")
    afrr = _validate_array_length(out.get("afrr_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "aFRR discharge for cycle cap")
    wh_before = wh.copy()
    afrr_before = afrr.copy()

    annual_cap = max(float(inputs.max_cycles_per_year), 0.0) * float(inputs.batt_energy_mwh)
    if annual_cap <= 1e-12:
        keep_wh = np.zeros(QH_PER_YEAR, dtype=float)
        keep_afrr = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        batt_sell = _validate_array_length(inputs.batt_sell_price, "BESS sell price for cycle cap")
        afrr_sell = _validate_array_length(inputs.afrr_discharge_price_qh if inputs.afrr_discharge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR UP price for cycle cap")
        stored_cost = np.nan_to_num(
            np.asarray(out.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)[:QH_PER_YEAR],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if len(stored_cost) != QH_PER_YEAR:
            stored_cost = np.zeros(QH_PER_YEAR, dtype=float)

        expected_up = np.zeros(QH_PER_YEAR, dtype=float)
        expected_cap_rev = np.zeros(QH_PER_YEAR, dtype=float)
        if afrr_capacity_result is not None:
            expected_up = np.asarray(afrr_capacity_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)
            expected_cap_rev = np.asarray(afrr_capacity_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)
            if len(expected_up) != QH_PER_YEAR:
                expected_up = np.zeros(QH_PER_YEAR, dtype=float)
            if len(expected_cap_rev) != QH_PER_YEAR:
                expected_cap_rev = np.zeros(QH_PER_YEAR, dtype=float)
        cap_value_per_activated_mwh = np.divide(
            expected_cap_rev,
            np.maximum(expected_up, 1e-12),
            out=np.zeros(QH_PER_YEAR, dtype=float),
            where=expected_up > 1e-12,
        )

        cost_per_output = stored_cost / max(inputs.eta_discharge, 1e-12)
        wh_value = batt_sell - cost_per_output - (float(inputs.cycle_cost_eur_per_mwh) / max(inputs.eta_discharge, 1e-12))
        afrr_value = afrr_sell + cap_value_per_activated_mwh - cost_per_output - (float(inputs.afrr_cycle_cost_eur_per_mwh) / max(inputs.eta_discharge, 1e-12))

        candidates: list[tuple[float, int, str, float]] = []
        for t in range(QH_PER_YEAR):
            if wh[t] > 1e-12:
                candidates.append((float(wh_value[t]), t, "wholesale", float(wh[t])))
            if afrr[t] > 1e-12:
                candidates.append((float(afrr_value[t]), t, "afrr", float(afrr[t])))

        # Highest value first; stable tie-break by time for deterministic results.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        keep_wh = np.zeros(QH_PER_YEAR, dtype=float)
        keep_afrr = np.zeros(QH_PER_YEAR, dtype=float)
        remaining = float(annual_cap)
        for value, t, market, qty in candidates:
            if remaining <= 1e-9:
                break
            kept = min(qty, remaining)
            if kept <= 1e-12:
                continue
            if market == "wholesale":
                keep_wh[t] += kept
            else:
                keep_afrr[t] += kept
            remaining -= kept

    wh_rejected = np.maximum(wh_before - keep_wh, 0.0)
    afrr_rejected = np.maximum(afrr_before - keep_afrr, 0.0)

    out["wholesale_discharge_qh_mwh"] = keep_wh
    out["wholesale_discharge_hourly_mwh"] = keep_wh
    out["afrr_discharge_qh_mwh"] = keep_afrr
    out["afrr_discharge_hourly_mwh"] = keep_afrr

    # Remove aFRR UP activation flags where no aFRR discharge remains.
    if "afrr_energy_up_activated_qh" in out:
        out["afrr_energy_up_activated_qh"] = (keep_afrr > 1e-12).astype(int)
    if "afrr_energy_up_activated_hourly" in out:
        out["afrr_energy_up_activated_hourly"] = (keep_afrr > 1e-12).astype(int)

    # Recompute revenues and SOC after the budget cut. Charges are clipped by
    # headroom during this SOC replay, so the final trajectory remains feasible.
    batt_sell = _validate_array_length(inputs.batt_sell_price, "BESS sell price for post cap")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Grid buy price for post cap")
    afrr_charge_price = _validate_array_length(inputs.afrr_charge_price_qh if inputs.afrr_charge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR DOWN price for post cap")
    afrr_discharge_price = _validate_array_length(inputs.afrr_discharge_price_qh if inputs.afrr_discharge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR UP price for post cap")

    pv_to_batt = _validate_array_length(out.get("wholesale_pv_to_batt_qh_mwh", np.zeros(QH_PER_YEAR)), "PV to battery post cap")
    pv_curt_to_batt = _validate_array_length(out.get("wholesale_pv_curtailed_to_batt_qh_mwh", np.zeros(QH_PER_YEAR)), "Curtailed PV to battery post cap")
    grid_charge = _validate_array_length(out.get("wholesale_grid_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "Grid charge post cap")
    afrr_charge = _validate_array_length(out.get("afrr_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "aFRR charge post cap")

    min_soc = float(inputs.batt_energy_mwh) * float(inputs.min_soc_pct) / 100.0
    max_soc = float(inputs.batt_energy_mwh) * float(inputs.max_soc_pct) / 100.0
    soc = np.zeros(QH_PER_YEAR + 1, dtype=float)
    soc[0] = min(max(float(inputs.initial_soc_mwh), min_soc), max_soc)

    for t in range(QH_PER_YEAR):
        total_charge_input = pv_to_batt[t] + pv_curt_to_batt[t] + grid_charge[t] + afrr_charge[t]
        max_charge_input = max(max_soc - soc[t], 0.0) / max(inputs.eta_charge, 1e-12)
        if total_charge_input > max_charge_input + 1e-12:
            scale = max_charge_input / max(total_charge_input, 1e-12)
            pv_to_batt[t] *= scale
            pv_curt_to_batt[t] *= scale
            grid_charge[t] *= scale
            afrr_charge[t] *= scale
            total_charge_input = max_charge_input
        total_discharge = keep_wh[t] + keep_afrr[t]
        max_discharge = max(soc[t] - min_soc, 0.0) * max(inputs.eta_discharge, 1e-12)
        if total_discharge > max_discharge + 1e-12:
            scale = max_discharge / max(total_discharge, 1e-12)
            keep_wh[t] *= scale
            keep_afrr[t] *= scale
            total_discharge = max_discharge
        soc[t + 1] = min(max(soc[t] + total_charge_input * inputs.eta_charge - total_discharge / max(inputs.eta_discharge, 1e-12), min_soc), max_soc)

    out["wholesale_pv_to_batt_qh_mwh"] = pv_to_batt
    out["wholesale_pv_to_batt_hourly_mwh"] = pv_to_batt
    out["wholesale_pv_curtailed_to_batt_qh_mwh"] = pv_curt_to_batt
    out["wholesale_pv_curtailed_to_batt_hourly_mwh"] = pv_curt_to_batt
    out["wholesale_grid_charge_qh_mwh"] = grid_charge
    out["wholesale_grid_charge_hourly_mwh"] = grid_charge
    out["afrr_charge_qh_mwh"] = afrr_charge
    out["afrr_charge_hourly_mwh"] = afrr_charge
    out["wholesale_discharge_qh_mwh"] = keep_wh
    out["wholesale_discharge_hourly_mwh"] = keep_wh
    out["afrr_discharge_qh_mwh"] = keep_afrr
    out["afrr_discharge_hourly_mwh"] = keep_afrr

    out["combined_soc_qh"] = soc
    out["combined_soc_hourly_end_mwh"] = soc[1:]
    out["combined_charge_to_soc_qh_mwh"] = (pv_to_batt + pv_curt_to_batt + grid_charge + afrr_charge) * inputs.eta_charge
    out["combined_discharge_from_soc_qh_mwh"] = (keep_wh + keep_afrr) / max(inputs.eta_discharge, 1e-12)

    out["wholesale_batt_sale_revenue_qh_eur"] = keep_wh * batt_sell
    out["wholesale_batt_sale_revenue_hourly_eur"] = out["wholesale_batt_sale_revenue_qh_eur"]
    out["wholesale_grid_charge_cost_qh_eur"] = grid_charge * grid_buy
    out["wholesale_grid_charge_cost_hourly_eur"] = out["wholesale_grid_charge_cost_qh_eur"]
    out["afrr_charge_cost_qh_eur"] = afrr_charge * afrr_charge_price
    out["afrr_charge_cost_hourly_eur"] = out["afrr_charge_cost_qh_eur"]
    out["afrr_sale_revenue_qh_eur"] = keep_afrr * afrr_discharge_price
    out["afrr_sale_revenue_hourly_eur"] = out["afrr_sale_revenue_qh_eur"]
    out["afrr_cycle_cost_qh_eur"] = keep_afrr / max(inputs.eta_discharge, 1e-12) * inputs.afrr_cycle_cost_eur_per_mwh
    out["afrr_cycle_cost_hourly_eur"] = out["afrr_cycle_cost_qh_eur"]
    out["afrr_net_revenue_qh_eur"] = out["afrr_sale_revenue_qh_eur"] - out["afrr_charge_cost_qh_eur"]
    out["afrr_net_revenue_hourly_eur"] = out["afrr_net_revenue_qh_eur"]

    combined_discharge = keep_wh + keep_afrr
    cumulative = np.cumsum(combined_discharge)
    remaining = np.maximum(annual_cap - cumulative, 0.0)
    out["annual_discharge_cap_mwh"] = np.full(QH_PER_YEAR, annual_cap, dtype=float)
    out["cumulative_battery_discharge_mwh"] = cumulative
    out["remaining_discharge_budget_mwh"] = remaining
    out["cycle_budget_used_pct"] = np.divide(cumulative, max(annual_cap, 1e-12), out=np.zeros(QH_PER_YEAR), where=annual_cap > 1e-12) * 100.0
    out["cycle_budget_available_flag"] = (remaining > 1e-9).astype(int)
    out["wholesale_discharge_rejected_due_to_cycle_budget"] = (wh_rejected > 1e-9).astype(int)
    out["afrr_up_discharge_rejected_due_to_cycle_budget"] = (afrr_rejected > 1e-9).astype(int)
    out["afrr_up_capacity_rejected_due_to_cycle_budget"] = (afrr_rejected > 1e-9).astype(int)
    out["discharge_rejected_due_to_cycle_budget"] = ((wh_rejected + afrr_rejected) > 1e-9).astype(int)

    # Ranking audit: 1 is highest net value among selected/rejected discharge candidates.
    net_value = np.zeros(QH_PER_YEAR, dtype=float)
    rank = np.zeros(QH_PER_YEAR, dtype=int)
    try:
        stored_cost = np.nan_to_num(np.asarray(out.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)[:QH_PER_YEAR], nan=0.0)
        cost_per_output = stored_cost / max(inputs.eta_discharge, 1e-12)
        wh_value = batt_sell - cost_per_output - inputs.cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        afrr_value = afrr_discharge_price - cost_per_output - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        net_value = np.maximum(np.where((keep_wh + wh_rejected) > 1e-12, wh_value, -1e30), np.where((keep_afrr + afrr_rejected) > 1e-12, afrr_value, -1e30))
        candidate_idx = np.where(net_value > -1e20)[0]
        order = candidate_idx[np.argsort(-net_value[candidate_idx])]
        rank[order] = np.arange(1, len(order) + 1)
        net_value[net_value <= -1e20] = 0.0
    except Exception:
        net_value = np.zeros(QH_PER_YEAR, dtype=float)
        rank = np.zeros(QH_PER_YEAR, dtype=int)
    out["net_dispatch_value_eur_per_mwh"] = net_value
    out["cycle_budget_rank"] = rank

    return out, {
        "wholesale_rejected": int(np.sum(wh_rejected > 1e-9)),
        "afrr_rejected": int(np.sum(afrr_rejected > 1e-9)),
    }

def build_final_result_after_market_arbitration(
    base_result: Dict[str, np.ndarray],
    reconciliation: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    final = dict(base_result)

    final["pv_to_batt"] = reconciliation["wholesale_pv_to_batt_hourly_mwh"]
    final["grid_charge"] = reconciliation["wholesale_grid_charge_hourly_mwh"]
    final["pv_curtailed_to_battery"] = reconciliation[
        "wholesale_pv_curtailed_to_batt_hourly_mwh"
    ]
    final["discharge"] = reconciliation["wholesale_discharge_hourly_mwh"]
    final["batt_sale_revenue"] = reconciliation["wholesale_batt_sale_revenue_hourly_eur"]
    final["grid_charge_cost"] = reconciliation["wholesale_grid_charge_cost_hourly_eur"]
    # Option A: cycle cost is kept as a theoretical degradation metric only; it is not deducted from cash revenue.
    final["wholesale_cycle_cost_eur"] = final["discharge"] * inputs.cycle_cost_eur_per_mwh

    total_batt_sale_revenue = float(final["batt_sale_revenue"].sum())
    total_grid_charge_cost = float(final["grid_charge_cost"].sum())
    total_wholesale_cycle_cost = float(final["wholesale_cycle_cost_eur"].sum())
    total_direct_pv_revenue = float(final["pv_direct_revenue"].sum())
    nightly_revenue_total = float(final["nightly_revenue_total"][0])

    total_discharged_mwh = float(final["discharge"].sum() + reconciliation["afrr_discharge_hourly_mwh"].sum())
    annual_discharge_cap_mwh = float(inputs.max_cycles_per_year) * float(inputs.batt_energy_mwh)

    final["total_batt_sale_revenue"] = np.array([total_batt_sale_revenue])
    final["total_grid_charge_cost"] = np.array([total_grid_charge_cost])
    final["total_wholesale_cycle_cost_eur"] = np.array([total_wholesale_cycle_cost])
    final["gross_bess_revenue_before_cycle_cost_eur"] = np.array([total_batt_sale_revenue])
    final["net_bess_revenue_after_cycle_cost_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])
    final["bess_cash_revenue_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])
    final["energy_shifted_mwh"] = np.array([total_discharged_mwh])
    final["energy_sold_total_mwh"] = np.array([float(final["pv_direct"].sum() + total_discharged_mwh)])
    final["equivalent_cycles"] = np.array([float(total_discharged_mwh / max(inputs.batt_energy_mwh, 1e-12))])
    final["max_cycles_per_year"] = np.array([float(inputs.max_cycles_per_year)])
    final["annual_discharge_cap_mwh"] = np.array([annual_discharge_cap_mwh])
    final["remaining_cycle_budget_mwh"] = np.array([max(annual_discharge_cap_mwh - total_discharged_mwh, 0.0)])
    if float(final["equivalent_cycles"][0]) > float(inputs.max_cycles_per_year) + 1e-6:
        raise RuntimeError(
            "Annual cycle cap exceeded after final dispatch reconciliation: "
            f"{float(final['equivalent_cycles'][0]):.6f} cycles > {float(inputs.max_cycles_per_year):.6f} cycles."
        )

    final["total_revenue"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])

    final["afrr_charge_hourly_mwh"] = reconciliation["afrr_charge_hourly_mwh"]
    final["afrr_discharge_hourly_mwh"] = reconciliation["afrr_discharge_hourly_mwh"]
    final["afrr_charge_cost_hourly_eur"] = reconciliation["afrr_charge_cost_hourly_eur"]
    final["afrr_sale_revenue_hourly_eur"] = reconciliation["afrr_sale_revenue_hourly_eur"]
    final["afrr_cycle_cost_hourly_eur"] = reconciliation["afrr_cycle_cost_hourly_eur"]
    final["afrr_net_revenue_hourly_eur"] = final["afrr_sale_revenue_hourly_eur"] - final["afrr_charge_cost_hourly_eur"]
    for _audit_key in [
        "stored_energy_cost_eur_per_mwh",
        "effective_discharge_value_eur_per_mwh",
        "spread_condition_respected",
        "wholesale_discharge_spread_ok",
        "afrr_up_discharge_spread_ok",
        "annual_discharge_cap_mwh",
        "cumulative_battery_discharge_mwh",
        "remaining_discharge_budget_mwh",
        "cycle_budget_used_pct",
        "cycle_budget_available_flag",
        "discharge_rejected_due_to_cycle_budget",
        "wholesale_discharge_rejected_due_to_cycle_budget",
        "afrr_up_discharge_rejected_due_to_cycle_budget",
        "afrr_up_capacity_rejected_due_to_cycle_budget",
        "net_dispatch_value_eur_per_mwh",
        "cycle_budget_rank",
    ]:
        if _audit_key in reconciliation:
            final[_audit_key] = reconciliation[_audit_key]

    final["total_afrr_charge_cost_eur"] = np.array([float(reconciliation["afrr_charge_cost_hourly_eur"].sum())])
    final["total_afrr_sale_revenue_eur"] = np.array([float(reconciliation["afrr_sale_revenue_hourly_eur"].sum())])
    final["total_afrr_cycle_cost_eur"] = np.array([float(reconciliation["afrr_cycle_cost_hourly_eur"].sum())])
    final["total_afrr_net_revenue_eur"] = np.array([float(final["afrr_net_revenue_hourly_eur"].sum())])

    final["total_battery_revenue_including_afrr_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(final["afrr_net_revenue_hourly_eur"].sum())
    ])

    final["total_revenue_including_afrr_eur"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(final["afrr_net_revenue_hourly_eur"].sum())
    ])

    return final

def add_afrr_capacity_to_final_result(
    result: Dict[str, np.ndarray],
    afrr_capacity_result: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Attach aFRR Capacity hourly arrays and revenue totals to the result dict."""
    final = dict(result)

    if afrr_capacity_result is None:
        up_revenue = np.zeros(QH_PER_YEAR, dtype=float)
        down_revenue = np.zeros(QH_PER_YEAR, dtype=float)
        total_revenue_h = np.zeros(QH_PER_YEAR, dtype=float)
        up_awarded = np.zeros(QH_PER_YEAR, dtype=int)
        down_awarded = np.zeros(QH_PER_YEAR, dtype=int)
        selected_market = np.full(QH_PER_YEAR, "none", dtype=object)
        certified_up_h = np.zeros(QH_PER_YEAR, dtype=float)
        certified_down_h = np.zeros(QH_PER_YEAR, dtype=float)
        eligible_h = np.zeros(QH_PER_YEAR, dtype=int)
    else:
        up_revenue = np.asarray(afrr_capacity_result["afrr_capacity_up_revenue_h_eur"], dtype=float)
        down_revenue = np.asarray(afrr_capacity_result["afrr_capacity_down_revenue_h_eur"], dtype=float)
        total_revenue_h = np.asarray(afrr_capacity_result["afrr_capacity_total_revenue_h_eur"], dtype=float)
        up_awarded = np.asarray(afrr_capacity_result["afrr_capacity_up_awarded_h"], dtype=int)
        down_awarded = np.asarray(afrr_capacity_result["afrr_capacity_down_awarded_h"], dtype=int)
        selected_market = np.asarray(afrr_capacity_result["afrr_capacity_selected_market_h"], dtype=object)
        certified_up_h = np.asarray(afrr_capacity_result["afrr_certified_capacity_up_mw_h"], dtype=float)
        certified_down_h = np.asarray(afrr_capacity_result["afrr_certified_capacity_down_mw_h"], dtype=float)
        eligible_h = np.asarray(afrr_capacity_result["afrr_capacity_eligible_h"], dtype=int)

    cap_up_total = float(up_revenue.sum())
    cap_down_total = float(down_revenue.sum())
    cap_total = float(total_revenue_h.sum())

    afrr_energy_net = float(final["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in final else 0.0
    base_battery_revenue = (
        float(final["total_batt_sale_revenue"][0])
        - float(final["total_grid_charge_cost"][0])
        + float(final["nightly_revenue_total"][0])
    )
    total_direct_pv_revenue = float(final["total_direct_pv_revenue"][0])

    final["afrr_capacity_up_revenue_h_eur"] = up_revenue
    final["afrr_capacity_down_revenue_h_eur"] = down_revenue
    final["afrr_capacity_total_revenue_h_eur"] = total_revenue_h
    final["afrr_capacity_up_awarded_h"] = up_awarded
    final["afrr_capacity_down_awarded_h"] = down_awarded
    final["afrr_capacity_selected_market_h"] = selected_market
    final["afrr_capacity_eligible_h"] = eligible_h
    final["afrr_certified_capacity_up_mw_h"] = certified_up_h
    final["afrr_certified_capacity_down_mw_h"] = certified_down_h

    if afrr_capacity_result is not None:
        for _k in [
            "wholesale_opportunity_value_eur",
            "wholesale_expected_value_after_capture_rate_eur",
            "raw_up_capacity_revenue_eur",
            "expected_up_capacity_revenue_eur",
            "raw_down_capacity_revenue_eur",
            "expected_down_capacity_revenue_eur",
            "expected_up_activated_mwh",
            "expected_down_activated_mwh",
            "afrr_up_energy_expected_value_eur",
            "afrr_down_energy_expected_value_eur",
            "afrr_up_total_expected_value_eur",
            "afrr_down_total_expected_value_eur",
            "selected_market",
            "selected_capacity_direction",
            "afrr_capacity_success_rate_pct",
            "bess_wholesale_capture_rate_pct",
            "afrr_up_activation_pct",
            "afrr_down_activation_pct",
            "available_export_headroom_mwh",
            "available_soc_headroom_mwh",
            "available_discharge_from_soc_mwh",
            "required_up_soc_reserve_mwh",
            "required_down_soc_headroom_mwh",
            "expected_degradation_cost_eur",
            "future_best_market_value_eur_per_mwh",
            "future_best_market_type",
            "cross_market_spread_eur_per_mwh",
            "required_min_spread_eur_per_mwh",
            "spread_condition_respected",
            "charge_reason",
            "discharge_reason",
            "stored_energy_cost_eur_per_mwh",
            "effective_discharge_value_eur_per_mwh",
            "future_expected_afrr_up_value_eur",
            "future_expected_wholesale_value_eur",
            "future_expected_best_discharge_market",
            "wholesale_charge_for_future_afrr_flag",
            "afrr_down_charge_for_future_wholesale_flag",
            "afrr_down_charge_for_future_afrr_up_flag",
            "wholesale_discharge_spread_ok",
            "afrr_up_discharge_spread_ok",
            "forward_horizon_hours",
            "future_opportunity_selected",
            "forward_soc_before_capacity_selection_mwh",
            "forward_soc_after_capacity_selection_mwh",
            "afrr_up_soc_feasible",
            "afrr_down_soc_feasible",
            "afrr_up_rejected_due_to_soc",
            "afrr_down_rejected_due_to_soc",
            "afrr_up_expected_vs_actual_shortfall_mwh",
            "afrr_down_expected_vs_actual_shortfall_mwh",
            "afrr_up_rejected_due_to_final_combined_soc",
            "afrr_down_rejected_due_to_final_combined_soc",
            "afrr_optimization_method",
            "afrr_method_note",
            "afrr_block_id_4h",
            "afrr_block_start_qh",
            "afrr_block_end_qh_exclusive",
            "afrr_afry_block_best_market",
            "afrr_afry_rejection_reason",
            "afrr_afry_block_up_value_eur",
            "afrr_afry_block_down_value_eur",
            "afrr_afry_block_wholesale_value_eur",
            "afrr_milp_block_status",
            "afrr_milp_rejection_reason",
            "afrr_milp_binary_up_award",
            "afrr_milp_binary_down_award",
        ]:
            final[_k] = np.asarray(
                afrr_capacity_result.get(
                    _k,
                    np.full(QH_PER_YEAR, "none", dtype=object)
                    if _k in (
                        "selected_market",
                        "selected_capacity_direction",
                        "future_best_market_type",
                        "charge_reason",
                        "discharge_reason",
                        "future_expected_best_discharge_market",
                        "afrr_optimization_method",
                        "afrr_method_note",
                        "afrr_afry_block_best_market",
                        "afrr_afry_rejection_reason",
                        "afrr_milp_block_status",
                        "afrr_milp_rejection_reason",
                    )
                    else np.zeros(QH_PER_YEAR),
                ),
                dtype=object
                if _k in (
                    "selected_market",
                    "selected_capacity_direction",
                    "future_best_market_type",
                    "charge_reason",
                    "discharge_reason",
                    "future_expected_best_discharge_market",
                    "afrr_optimization_method",
                    "afrr_method_note",
                    "afrr_afry_block_best_market",
                    "afrr_afry_rejection_reason",
                    "afrr_milp_block_status",
                    "afrr_milp_rejection_reason",
                )
                else float,
            )

    final["total_afrr_capacity_up_revenue_eur"] = np.array([cap_up_total])
    final["total_afrr_capacity_down_revenue_eur"] = np.array([cap_down_total])
    final["total_afrr_capacity_revenue_eur"] = np.array([cap_total])

    final["total_battery_revenue_including_afrr_capacity_eur"] = np.array([
        base_battery_revenue + afrr_energy_net + cap_total
    ])
    final["total_revenue_including_afrr_capacity_eur"] = np.array([
        total_direct_pv_revenue + base_battery_revenue + afrr_energy_net + cap_total
    ])

    return final
