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
from optimization.forward_curves import compute_forward_cross_market_value_curves
from utils.time_utils import build_quarter_hour_index

def _hour_window_mask(start_hour: int, end_hour: int, length: int = QH_PER_YEAR) -> np.ndarray:
    """Return a quarter-hour eligibility mask for an hour window, including overnight windows."""
    idx = build_quarter_hour_index(DEFAULT_YEAR)
    hours = idx.hour.to_numpy()
    start = int(start_hour) % 24
    end = int(end_hour) % 24
    if start == end:
        mask = np.ones(len(idx), dtype=bool)
    elif start < end:
        mask = (hours >= start) & (hours < end)
    else:
        mask = (hours >= start) | (hours < end)
    return mask[:length]


def simulate_afrr_capacity(
    inputs: SimulationInputs,
    wholesale_reference_result: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Phase 1 aFRR capacity co-optimization at 15-minute resolution.

    Exclusive market selection: for each 15-minute timestep the battery chooses
    either wholesale, aFRR UP capacity + expected UP energy, aFRR DOWN capacity +
    expected DOWN energy, or no battery market action.

    This version is forward-SOC-aware: aFRR UP/DOWN capacity is selected only if
    a sequential SOC trajectory can deliver/absorb the expected activation MWh
    after accounting for previously selected aFRR capacity actions.
    """
    zero_f = np.zeros(QH_PER_YEAR, dtype=float)
    zero_i = np.zeros(QH_PER_YEAR, dtype=int)
    none_o = np.full(QH_PER_YEAR, "none", dtype=object)

    audit_zero_bool = np.zeros(QH_PER_YEAR, dtype=int)
    base_return = {
        "afrr_capacity_up_awarded_h": zero_i.copy(),
        "afrr_capacity_down_awarded_h": zero_i.copy(),
        "afrr_capacity_selected_market_h": none_o.copy(),
        "afrr_capacity_up_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_down_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_total_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_eligible_h": zero_i.copy(),
        "afrr_certified_capacity_up_mw_h": zero_f.copy(),
        "afrr_certified_capacity_down_mw_h": zero_f.copy(),
        "wholesale_opportunity_value_eur": zero_f.copy(),
        "wholesale_expected_value_after_capture_rate_eur": zero_f.copy(),
        "raw_up_capacity_revenue_eur": zero_f.copy(),
        "expected_up_capacity_revenue_eur": zero_f.copy(),
        "raw_down_capacity_revenue_eur": zero_f.copy(),
        "expected_down_capacity_revenue_eur": zero_f.copy(),
        "expected_up_activated_mwh": zero_f.copy(),
        "expected_down_activated_mwh": zero_f.copy(),
        "afrr_up_energy_expected_value_eur": zero_f.copy(),
        "afrr_down_energy_expected_value_eur": zero_f.copy(),
        "afrr_up_total_expected_value_eur": zero_f.copy(),
        "afrr_down_total_expected_value_eur": zero_f.copy(),
        "selected_market": none_o.copy(),
        "selected_capacity_direction": none_o.copy(),
        "afrr_capacity_success_rate_pct": np.full(QH_PER_YEAR, float(inputs.afrr_capacity_success_rate_pct), dtype=float),
        "bess_wholesale_capture_rate_pct": np.full(QH_PER_YEAR, float(inputs.bess_capture_rate_pct), dtype=float),
        "afrr_up_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_up_activation_pct), dtype=float),
        "afrr_down_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_down_activation_pct), dtype=float),
        "available_export_headroom_mwh": zero_f.copy(),
        "available_soc_headroom_mwh": zero_f.copy(),
        "available_discharge_from_soc_mwh": zero_f.copy(),
        "required_up_soc_reserve_mwh": zero_f.copy(),
        "required_down_soc_headroom_mwh": zero_f.copy(),
        "expected_degradation_cost_eur": zero_f.copy(),
        "future_best_market_value_eur_per_mwh": zero_f.copy(),
        "future_best_market_type": none_o.copy(),
        "cross_market_spread_eur_per_mwh": zero_f.copy(),
        "required_min_spread_eur_per_mwh": zero_f.copy(),
        "spread_condition_respected": audit_zero_bool.copy(),
        "charge_reason": none_o.copy(),
        "discharge_reason": none_o.copy(),
        "stored_energy_cost_eur_per_mwh": zero_f.copy(),
        "effective_discharge_value_eur_per_mwh": zero_f.copy(),
        "future_expected_afrr_up_value_eur": zero_f.copy(),
        "future_expected_wholesale_value_eur": zero_f.copy(),
        "future_expected_best_discharge_market": none_o.copy(),
        "wholesale_charge_for_future_afrr_flag": audit_zero_bool.copy(),
        "afrr_down_charge_for_future_wholesale_flag": audit_zero_bool.copy(),
        "afrr_down_charge_for_future_afrr_up_flag": audit_zero_bool.copy(),
        "wholesale_discharge_spread_ok": audit_zero_bool.copy(),
        "afrr_up_discharge_spread_ok": audit_zero_bool.copy(),
        "forward_horizon_hours": zero_f.copy(),
        "future_opportunity_selected": audit_zero_bool.copy(),
        "forward_soc_before_capacity_selection_mwh": zero_f.copy(),
        "forward_soc_after_capacity_selection_mwh": zero_f.copy(),
        "afrr_up_soc_feasible": audit_zero_bool.copy(),
        "afrr_down_soc_feasible": audit_zero_bool.copy(),
        "afrr_up_rejected_due_to_soc": audit_zero_bool.copy(),
        "afrr_down_rejected_due_to_soc": audit_zero_bool.copy(),
        "afrr_up_expected_vs_actual_shortfall_mwh": zero_f.copy(),
        "afrr_down_expected_vs_actual_shortfall_mwh": zero_f.copy(),
        "afrr_up_rejected_due_to_final_combined_soc": audit_zero_bool.copy(),
        "afrr_down_rejected_due_to_final_combined_soc": audit_zero_bool.copy(),
    }

    if not inputs.enable_afrr_capacity:
        return base_return
    if inputs.afrr_capacity_up_price_h is None or inputs.afrr_capacity_down_price_h is None:
        raise ValueError("Les deux courbes aFRR Capacity UP et Down doivent être fournies.")

    up_price = _validate_array_length(inputs.afrr_capacity_up_price_h, "Prix aFRR Capacity UP", QH_PER_YEAR)
    down_price = _validate_array_length(inputs.afrr_capacity_down_price_h, "Prix aFRR Capacity Down", QH_PER_YEAR)

    if inputs.afrr_charge_price_qh is None:
        afrr_down_energy_price = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        afrr_down_energy_price = _validate_array_length(inputs.afrr_charge_price_qh, "Prix aFRR Down Energy", QH_PER_YEAR)
    if inputs.afrr_discharge_price_qh is None:
        afrr_up_energy_price = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        afrr_up_energy_price = _validate_array_length(inputs.afrr_discharge_price_qh, "Prix aFRR Up Energy", QH_PER_YEAR)

    if not (0.0 <= inputs.afrr_certified_capacity_pct <= 100.0):
        raise ValueError("% of Certified Capacity for aFRR doit être compris entre 0 et 100 %.")
    if not (0.0 <= inputs.afrr_capacity_success_rate_pct <= 100.0):
        raise ValueError("aFRR Capacity Bid Success Rate (%) doit être compris entre 0 et 100 %.")

    success = float(inputs.afrr_capacity_success_rate_pct) / 100.0
    activation_up = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
    activation_down = min(max(float(inputs.afrr_energy_down_activation_pct) / 100.0, 0.0), 1.0)

    certified_up = float(inputs.afrr_certified_capacity_up_mw)
    certified_down = float(inputs.afrr_certified_capacity_down_mw)

    pv_direct = np.zeros(QH_PER_YEAR, dtype=float)
    wholesale_opportunity = np.zeros(QH_PER_YEAR, dtype=float)
    baseline_soc = np.full(QH_PER_YEAR + 1, float(inputs.initial_soc_mwh), dtype=float)
    baseline_soc_delta = np.zeros(QH_PER_YEAR, dtype=float)

    if wholesale_reference_result is not None:
        pv_direct = _validate_array_length(wholesale_reference_result.get("pv_direct", pv_direct), "PV direct référence wholesale", QH_PER_YEAR)
        soc_curve = np.asarray(wholesale_reference_result.get("soc", baseline_soc), dtype=float).reshape(-1)
        if len(soc_curve) >= QH_PER_YEAR + 1:
            baseline_soc = soc_curve[:QH_PER_YEAR + 1].astype(float)
            baseline_soc_delta = np.diff(baseline_soc)
        batt_sale = np.asarray(wholesale_reference_result.get("batt_sale_revenue", zero_f), dtype=float).reshape(-1)
        pv_to_batt = np.asarray(wholesale_reference_result.get("pv_to_batt", zero_f), dtype=float).reshape(-1)
        curtailed_to_batt = np.asarray(wholesale_reference_result.get("pv_curtailed_to_battery", zero_f), dtype=float).reshape(-1)
        discharge = np.asarray(wholesale_reference_result.get("discharge", zero_f), dtype=float).reshape(-1)
        future_best_sell = np.maximum.accumulate(np.asarray(inputs.batt_sell_price, dtype=float)[::-1])[::-1]
        charge_future_value = (pv_to_batt + curtailed_to_batt) * np.maximum(future_best_sell * inputs.eta_charge * inputs.eta_discharge - inputs.pv_price, 0.0)
        discharge_value = np.maximum(batt_sale - discharge * inputs.cycle_cost_eur_per_mwh, 0.0)
        grid_charge_value = np.maximum((future_best_sell * inputs.eta_charge * inputs.eta_discharge - inputs.grid_buy_price) * np.asarray(wholesale_reference_result.get("grid_charge", zero_f), dtype=float), 0.0)
        wholesale_opportunity = np.maximum.reduce([discharge_value, charge_future_value, grid_charge_value, np.zeros(QH_PER_YEAR)])

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    export_limit_qh = inputs.grid_export_limit_mw * QH_DT_HOURS
    available_export_headroom = np.maximum(export_limit_qh - pv_direct, 0.0)

    required_up_soc_reserve_mwh = certified_up * QH_DT_HOURS * activation_up
    required_down_soc_headroom_mwh = certified_down * QH_DT_HOURS * activation_down

    raw_up_capacity = up_price * certified_up * QH_DT_HOURS
    raw_down_capacity = down_price * certified_down * QH_DT_HOURS
    expected_up_capacity = raw_up_capacity * success
    expected_down_capacity = raw_down_capacity * success
    wholesale_expected = np.maximum(wholesale_opportunity, 0.0)

    forward_curves = compute_forward_cross_market_value_curves(inputs)
    future_best_value = forward_curves["future_best_market_value_eur_per_mwh"]
    future_best_type = forward_curves["future_best_market_type"]
    future_wholesale_value = forward_curves["future_expected_wholesale_value_eur_per_mwh"]
    future_afrr_up_value = forward_curves["future_expected_afrr_up_value_eur_per_mwh"]

    stored_energy_cost_qh = np.zeros(QH_PER_YEAR, dtype=float)
    if wholesale_reference_result is not None and "avg_stored_charge_price" in wholesale_reference_result:
        avg_cost = np.asarray(wholesale_reference_result["avg_stored_charge_price"], dtype=float).reshape(-1)
        if len(avg_cost) >= QH_PER_YEAR:
            stored_energy_cost_qh = np.nan_to_num(avg_cost[:QH_PER_YEAR], nan=0.0, posinf=0.0, neginf=0.0)

    capacity_eligible_mask = _hour_window_mask(inputs.afrr_capacity_start_hour, inputs.afrr_capacity_end_hour, QH_PER_YEAR)

    selected = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_market = np.full(QH_PER_YEAR, "none", dtype=object)
    up_awarded = np.zeros(QH_PER_YEAR, dtype=int)
    down_awarded = np.zeros(QH_PER_YEAR, dtype=int)
    up_revenue = np.zeros(QH_PER_YEAR, dtype=float)
    down_revenue = np.zeros(QH_PER_YEAR, dtype=float)
    expected_up_activated = np.zeros(QH_PER_YEAR, dtype=float)
    expected_down_activated = np.zeros(QH_PER_YEAR, dtype=float)
    up_energy_value_selected = np.zeros(QH_PER_YEAR, dtype=float)
    down_energy_value_selected = np.zeros(QH_PER_YEAR, dtype=float)
    up_total_value_audit = np.zeros(QH_PER_YEAR, dtype=float)
    down_total_value_audit = np.zeros(QH_PER_YEAR, dtype=float)
    expected_degradation_selected = np.zeros(QH_PER_YEAR, dtype=float)
    cross_market_spread = np.zeros(QH_PER_YEAR, dtype=float)
    required_min_spread = np.zeros(QH_PER_YEAR, dtype=float)
    spread_condition_respected = np.zeros(QH_PER_YEAR, dtype=int)
    charge_reason = np.full(QH_PER_YEAR, "none", dtype=object)
    discharge_reason = np.full(QH_PER_YEAR, "none", dtype=object)
    effective_discharge_value = np.zeros(QH_PER_YEAR, dtype=float)
    wholesale_charge_for_future_afrr_flag = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_down_charge_for_future_wholesale_flag = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_down_charge_for_future_afrr_up_flag = np.zeros(QH_PER_YEAR, dtype=int)
    wholesale_discharge_spread_ok = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_up_discharge_spread_ok = np.zeros(QH_PER_YEAR, dtype=int)
    future_opportunity_selected = np.zeros(QH_PER_YEAR, dtype=int)

    forward_soc_before = np.zeros(QH_PER_YEAR, dtype=float)
    forward_soc_after = np.zeros(QH_PER_YEAR, dtype=float)
    forward_soc_mwh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    forward_soc_mwh[0] = min(max(float(inputs.initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    available_soc_headroom_input = np.zeros(QH_PER_YEAR, dtype=float)
    available_discharge_output = np.zeros(QH_PER_YEAR, dtype=float)
    up_soc_feasible = np.zeros(QH_PER_YEAR, dtype=int)
    down_soc_feasible = np.zeros(QH_PER_YEAR, dtype=int)
    up_rejected_due_to_soc = np.zeros(QH_PER_YEAR, dtype=int)
    down_rejected_due_to_soc = np.zeros(QH_PER_YEAR, dtype=int)

    for t in range(QH_PER_YEAR):
        soc_now = min(max(forward_soc_mwh[t], min_soc_mwh), max_soc_mwh)
        forward_soc_before[t] = soc_now

        if not capacity_eligible_mask[t]:
            if wholesale_expected[t] > 0:
                selected_market[t] = "wholesale"
                forward_soc_mwh[t + 1] = min(max(soc_now + baseline_soc_delta[t], min_soc_mwh), max_soc_mwh)
            else:
                forward_soc_mwh[t + 1] = soc_now
            forward_soc_after[t] = forward_soc_mwh[t + 1]
            continue

        headroom_input_t = max(max_soc_mwh - soc_now, 0.0) / max(inputs.eta_charge, 1e-12)
        discharge_output_t = max(soc_now - min_soc_mwh, 0.0) * inputs.eta_discharge
        available_soc_headroom_input[t] = headroom_input_t
        available_discharge_output[t] = discharge_output_t

        # Expected activated MWh used for the economic comparison and later physical dispatch.
        # UP also needs export headroom; DOWN only needs SOC headroom in this Phase-1 model.
        up_target_full = certified_up * QH_DT_HOURS * activation_up
        down_target_full = certified_down * QH_DT_HOURS * activation_down
        up_target_t = min(up_target_full, available_export_headroom[t])
        down_target_t = down_target_full

        up_feasible_t = (
            certified_up > 0
            and up_target_t > 1e-12
            and available_export_headroom[t] + 1e-12 >= up_target_t
            and discharge_output_t + 1e-12 >= up_target_t
        )
        down_feasible_t = (
            certified_down > 0
            and down_target_t > 1e-12
            and headroom_input_t + 1e-12 >= down_target_t
        )
        up_soc_feasible[t] = int(up_feasible_t)
        down_soc_feasible[t] = int(down_feasible_t)

        stored_cost_per_output_mwh_t = stored_energy_cost_qh[t] / max(inputs.eta_discharge, 1e-12)
        afrr_up_spread_t = (
            afrr_up_energy_price[t]
            - stored_cost_per_output_mwh_t
            - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        )
        up_spread_ok_t = afrr_up_spread_t + 1e-12 >= inputs.afrr_min_spread_eur_per_mwh
        afrr_up_discharge_spread_ok[t] = int(up_spread_ok_t)

        up_energy_value_t = up_target_t * afrr_up_energy_price[t]
        up_degradation_t = up_target_t / max(inputs.eta_discharge, 1e-12) * inputs.afrr_cycle_cost_eur_per_mwh
        up_total_t = expected_up_capacity[t] + up_energy_value_t - up_degradation_t
        if not up_spread_ok_t:
            up_total_t = -1e30
        if not up_feasible_t:
            if up_total_t > max(wholesale_expected[t], 0.0):
                up_rejected_due_to_soc[t] = 1
            up_total_t = -1e30

        # DOWN sign convention: positive DOWN price is a charging cost; negative price is revenue/benefit.
        # Add cross-market future value: DOWN charge now can be used later for wholesale or aFRR UP discharge.
        future_output_mwh_t = down_target_t * inputs.eta_charge * inputs.eta_discharge
        down_energy_value_t = -down_target_t * afrr_down_energy_price[t]
        down_future_value_t = 0.0
        down_required_spread_t = inputs.afrr_min_spread_eur_per_mwh
        down_spread_t = -1e30
        if future_best_type[t] == "wholesale":
            down_required_spread_t = inputs.afrr_down_to_wholesale_min_spread_eur_per_mwh
        elif future_best_type[t] == "afrr_up":
            down_required_spread_t = inputs.afrr_min_spread_eur_per_mwh
        if future_best_value[t] > -1e20 and down_target_t > 1e-12:
            input_cost_per_future_output = afrr_down_energy_price[t] / max(inputs.eta_charge * inputs.eta_discharge, 1e-12)
            down_spread_t = future_best_value[t] - input_cost_per_future_output - inputs.afrr_cycle_cost_eur_per_mwh
            if down_spread_t + 1e-12 >= down_required_spread_t:
                down_future_value_t = future_output_mwh_t * future_best_value[t]
                future_opportunity_selected[t] = 1
                if future_best_type[t] == "wholesale":
                    afrr_down_charge_for_future_wholesale_flag[t] = 1
                elif future_best_type[t] == "afrr_up":
                    afrr_down_charge_for_future_afrr_up_flag[t] = 1
        down_total_t = expected_down_capacity[t] + down_energy_value_t + down_future_value_t
        if not down_feasible_t:
            if down_total_t > max(wholesale_expected[t], 0.0):
                down_rejected_due_to_soc[t] = 1
            down_total_t = -1e30

        up_total_value_audit[t] = 0.0 if up_total_t <= -1e20 else up_total_t
        down_total_value_audit[t] = 0.0 if down_total_t <= -1e20 else down_total_t

        best_val = max(float(wholesale_expected[t]), float(up_total_t), float(down_total_t), 0.0)
        if up_total_t == best_val and up_total_t > 0 and up_total_t > wholesale_expected[t] + 1e-9:
            selected[t] = "up"
            selected_market[t] = "afrr_up_capacity"
            up_awarded[t] = 1
            up_revenue[t] = expected_up_capacity[t]
            expected_up_activated[t] = up_target_t
            up_energy_value_selected[t] = up_energy_value_t
            expected_degradation_selected[t] = up_degradation_t
            cross_market_spread[t] = afrr_up_spread_t
            required_min_spread[t] = inputs.afrr_min_spread_eur_per_mwh
            spread_condition_respected[t] = int(up_spread_ok_t)
            discharge_reason[t] = "afrr_up_capacity_activation_spread_ok"
            effective_discharge_value[t] = afrr_up_energy_price[t]
            forward_soc_mwh[t + 1] = min(max(soc_now - up_target_t / max(inputs.eta_discharge, 1e-12), min_soc_mwh), max_soc_mwh)
        elif down_total_t == best_val and down_total_t > 0 and down_total_t > wholesale_expected[t] + 1e-9:
            selected[t] = "down"
            selected_market[t] = "afrr_down_capacity"
            down_awarded[t] = 1
            down_revenue[t] = expected_down_capacity[t]
            expected_down_activated[t] = down_target_t
            down_energy_value_selected[t] = down_energy_value_t + down_future_value_t
            cross_market_spread[t] = down_spread_t if down_spread_t > -1e20 else 0.0
            required_min_spread[t] = down_required_spread_t
            spread_condition_respected[t] = int(down_spread_t + 1e-12 >= down_required_spread_t)
            charge_reason[t] = "afrr_down_charge_for_future_" + str(future_best_type[t])
            effective_discharge_value[t] = max(future_best_value[t], 0.0)
            forward_soc_mwh[t + 1] = min(max(soc_now + down_target_t * inputs.eta_charge, min_soc_mwh), max_soc_mwh)
        elif wholesale_expected[t] == best_val and wholesale_expected[t] > 0:
            selected_market[t] = "wholesale"
            # Apply the baseline wholesale SOC movement, but from the forward SOC state.
            forward_soc_mwh[t + 1] = min(max(soc_now + baseline_soc_delta[t], min_soc_mwh), max_soc_mwh)
        else:
            forward_soc_mwh[t + 1] = soc_now

        forward_soc_after[t] = forward_soc_mwh[t + 1]

    return {
        "afrr_capacity_up_awarded_h": up_awarded,
        "afrr_capacity_down_awarded_h": down_awarded,
        "afrr_capacity_selected_market_h": selected,
        "afrr_capacity_up_revenue_h_eur": up_revenue,
        "afrr_capacity_down_revenue_h_eur": down_revenue,
        "afrr_capacity_total_revenue_h_eur": up_revenue + down_revenue,
        "afrr_capacity_eligible_h": capacity_eligible_mask.astype(int),
        "afrr_certified_capacity_up_mw_h": np.full(QH_PER_YEAR, certified_up, dtype=float),
        "afrr_certified_capacity_down_mw_h": np.full(QH_PER_YEAR, certified_down, dtype=float),
        "wholesale_opportunity_value_eur": wholesale_opportunity,
        "wholesale_expected_value_after_capture_rate_eur": wholesale_expected,
        "raw_up_capacity_revenue_eur": raw_up_capacity,
        "expected_up_capacity_revenue_eur": expected_up_capacity,
        "raw_down_capacity_revenue_eur": raw_down_capacity,
        "expected_down_capacity_revenue_eur": expected_down_capacity,
        "expected_up_activated_mwh": expected_up_activated,
        "expected_down_activated_mwh": expected_down_activated,
        "afrr_up_energy_expected_value_eur": up_energy_value_selected,
        "afrr_down_energy_expected_value_eur": down_energy_value_selected,
        "afrr_up_total_expected_value_eur": up_total_value_audit,
        "afrr_down_total_expected_value_eur": down_total_value_audit,
        "selected_market": selected_market,
        "selected_capacity_direction": selected,
        "afrr_capacity_success_rate_pct": np.full(QH_PER_YEAR, float(inputs.afrr_capacity_success_rate_pct), dtype=float),
        "bess_wholesale_capture_rate_pct": np.full(QH_PER_YEAR, float(inputs.bess_capture_rate_pct), dtype=float),
        "afrr_up_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_up_activation_pct), dtype=float),
        "afrr_down_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_down_activation_pct), dtype=float),
        "available_export_headroom_mwh": available_export_headroom,
        "available_soc_headroom_mwh": available_soc_headroom_input,
        "available_discharge_from_soc_mwh": available_discharge_output,
        "required_up_soc_reserve_mwh": np.full(QH_PER_YEAR, required_up_soc_reserve_mwh, dtype=float),
        "required_down_soc_headroom_mwh": np.full(QH_PER_YEAR, required_down_soc_headroom_mwh, dtype=float),
        "expected_degradation_cost_eur": expected_degradation_selected,
        "future_best_market_value_eur_per_mwh": future_best_value,
        "future_best_market_type": future_best_type,
        "cross_market_spread_eur_per_mwh": cross_market_spread,
        "required_min_spread_eur_per_mwh": required_min_spread,
        "spread_condition_respected": spread_condition_respected,
        "charge_reason": charge_reason,
        "discharge_reason": discharge_reason,
        "stored_energy_cost_eur_per_mwh": stored_energy_cost_qh,
        "effective_discharge_value_eur_per_mwh": effective_discharge_value,
        "future_expected_afrr_up_value_eur": future_afrr_up_value,
        "future_expected_wholesale_value_eur": future_wholesale_value,
        "future_expected_best_discharge_market": future_best_type,
        "wholesale_charge_for_future_afrr_flag": wholesale_charge_for_future_afrr_flag,
        "afrr_down_charge_for_future_wholesale_flag": afrr_down_charge_for_future_wholesale_flag,
        "afrr_down_charge_for_future_afrr_up_flag": afrr_down_charge_for_future_afrr_up_flag,
        "wholesale_discharge_spread_ok": wholesale_discharge_spread_ok,
        "afrr_up_discharge_spread_ok": afrr_up_discharge_spread_ok,
        "forward_horizon_hours": forward_curves["forward_horizon_hours"],
        "future_opportunity_selected": future_opportunity_selected,
        "forward_soc_before_capacity_selection_mwh": forward_soc_before,
        "forward_soc_after_capacity_selection_mwh": forward_soc_after,
        "afrr_up_soc_feasible": up_soc_feasible,
        "afrr_down_soc_feasible": down_soc_feasible,
        "afrr_up_rejected_due_to_soc": up_rejected_due_to_soc,
        "afrr_down_rejected_due_to_soc": down_rejected_due_to_soc,
        "afrr_up_expected_vs_actual_shortfall_mwh": np.zeros(QH_PER_YEAR, dtype=float),
        "afrr_down_expected_vs_actual_shortfall_mwh": np.zeros(QH_PER_YEAR, dtype=float),
        "afrr_up_rejected_due_to_final_combined_soc": np.zeros(QH_PER_YEAR, dtype=int),
        "afrr_down_rejected_due_to_final_combined_soc": np.zeros(QH_PER_YEAR, dtype=int),
    }

def enforce_afrr_capacity_deliverability_from_final_dispatch(
    afrr_capacity_result: Dict[str, np.ndarray],
    reconciliation: Dict[str, np.ndarray] | None,
    tolerance_mwh: float = 1e-6,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Remove aFRR capacity awards that cannot be delivered in final dispatch.

    simulate_afrr_capacity() uses a forward-SOC approximation before the final
    wholesale/aFRR reconciliation is known. This post-pass validates awarded
    capacity against the actual final combined dispatch. Any UP award whose
    expected activation is not physically delivered in the final reconciliation
    is removed, then the caller should rerun final DP/aFRR dispatch using the
    filtered awards. This makes UP capacity awards require deliverability under
    the final combined SOC trajectory, not only the preliminary forward tracker.
    """
    if afrr_capacity_result is None or reconciliation is None:
        return afrr_capacity_result, {"removed_up": 0, "removed_down": 0}

    filtered: Dict[str, np.ndarray] = {}
    for key, value in afrr_capacity_result.items():
        if isinstance(value, np.ndarray):
            filtered[key] = value.copy()
        else:
            filtered[key] = value

    selected = np.asarray(filtered.get("afrr_capacity_selected_market_h", np.full(QH_PER_YEAR, "none", dtype=object)), dtype=object).copy()
    selected_market = np.asarray(filtered.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)), dtype=object).copy()
    selected_direction = np.asarray(filtered.get("selected_capacity_direction", selected), dtype=object).copy()

    expected_up = _validate_array_length(filtered.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)), "Expected aFRR UP activated MWh")
    expected_down = _validate_array_length(filtered.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)), "Expected aFRR DOWN activated MWh")
    actual_up = _validate_array_length(reconciliation.get("afrr_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "Actual aFRR UP discharge MWh")
    actual_down = _validate_array_length(reconciliation.get("afrr_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "Actual aFRR DOWN charge MWh")

    up_shortfall = np.maximum(expected_up - actual_up, 0.0)
    down_shortfall = np.maximum(expected_down - actual_down, 0.0)

    # UP deliverability is the critical issue observed in the simulations:
    # reject any UP award that final combined SOC/export constraints cannot activate.
    remove_up = (selected == "up") & (up_shortfall > tolerance_mwh)
    # Apply the same consistency rule to DOWN as a safety check; it usually removes few/no intervals.
    remove_down = (selected == "down") & (down_shortfall > tolerance_mwh)

    removed_up_count = int(np.sum(remove_up))
    removed_down_count = int(np.sum(remove_down))

    if removed_up_count == 0 and removed_down_count == 0:
        filtered["afrr_up_expected_vs_actual_shortfall_mwh"] = up_shortfall
        filtered["afrr_down_expected_vs_actual_shortfall_mwh"] = down_shortfall
        filtered["afrr_up_rejected_due_to_final_combined_soc"] = np.zeros(QH_PER_YEAR, dtype=int)
        filtered["afrr_down_rejected_due_to_final_combined_soc"] = np.zeros(QH_PER_YEAR, dtype=int)
        return filtered, {"removed_up": 0, "removed_down": 0}

    rejected_up_final = np.asarray(filtered.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)), dtype=int).copy()
    rejected_down_final = np.asarray(filtered.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)), dtype=int).copy()
    rejected_up_final[remove_up] = 1
    rejected_down_final[remove_down] = 1

    remove_any = remove_up | remove_down
    selected[remove_any] = "none"
    selected_market[remove_any] = "none"
    selected_direction[remove_any] = "none"

    filtered["afrr_capacity_selected_market_h"] = selected
    filtered["selected_market"] = selected_market
    filtered["selected_capacity_direction"] = selected_direction

    if "afrr_capacity_up_awarded_h" in filtered:
        arr = np.asarray(filtered["afrr_capacity_up_awarded_h"], dtype=int).copy()
        arr[remove_up] = 0
        filtered["afrr_capacity_up_awarded_h"] = arr
    if "afrr_capacity_down_awarded_h" in filtered:
        arr = np.asarray(filtered["afrr_capacity_down_awarded_h"], dtype=int).copy()
        arr[remove_down] = 0
        filtered["afrr_capacity_down_awarded_h"] = arr

    zero_when_removed_keys = [
        "afrr_capacity_up_revenue_h_eur",
        "expected_up_activated_mwh",
        "afrr_up_energy_expected_value_eur",
        "expected_degradation_cost_eur",
    ]
    for key in zero_when_removed_keys:
        if key in filtered:
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[remove_up] = 0.0
            filtered[key] = arr

    zero_down_keys = [
        "afrr_capacity_down_revenue_h_eur",
        "expected_down_activated_mwh",
        "afrr_down_energy_expected_value_eur",
    ]
    for key in zero_down_keys:
        if key in filtered:
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[remove_down] = 0.0
            filtered[key] = arr

    for key, mask in [
        ("afrr_up_total_expected_value_eur", remove_up),
        ("afrr_down_total_expected_value_eur", remove_down),
        ("expected_up_capacity_revenue_eur", remove_up),
        ("expected_down_capacity_revenue_eur", remove_down),
    ]:
        # Keep raw price/value columns for audit, but zero selected expected value columns on rejected awards.
        if key in filtered and key not in ("expected_up_capacity_revenue_eur", "expected_down_capacity_revenue_eur"):
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[mask] = 0.0
            filtered[key] = arr

    up_rev = np.asarray(filtered.get("afrr_capacity_up_revenue_h_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    down_rev = np.asarray(filtered.get("afrr_capacity_down_revenue_h_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    filtered["afrr_capacity_total_revenue_h_eur"] = up_rev + down_rev

    filtered["afrr_up_expected_vs_actual_shortfall_mwh"] = up_shortfall
    filtered["afrr_down_expected_vs_actual_shortfall_mwh"] = down_shortfall
    filtered["afrr_up_rejected_due_to_final_combined_soc"] = rejected_up_final
    filtered["afrr_down_rejected_due_to_final_combined_soc"] = rejected_down_final

    return filtered, {"removed_up": removed_up_count, "removed_down": removed_down_count}
