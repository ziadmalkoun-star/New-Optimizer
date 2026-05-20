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

def _select_best_daily_afrr_competing_blocks(
    charge_prices_day: np.ndarray,
    discharge_prices_day: np.ndarray,
    grid_buy_prices_day: np.ndarray,
    batt_sell_prices_day: np.ndarray,
    eligible_mask_day: np.ndarray,
    eta_charge: float,
    eta_discharge: float,
    afrr_cycle_cost_eur_per_mwh: float,
    afrr_min_spread_eur_per_mwh: float,
    n_qh: int,
) -> Dict[str, object]:
    """Mode 2 candidate selection: aFRR competes with wholesale routes on eligible QHs."""
    eligible = np.asarray(eligible_mask_day, dtype=bool)
    charge_candidate = np.where(eligible & (charge_prices_day < grid_buy_prices_day))[0]
    discharge_candidate = np.where(eligible & (discharge_prices_day > batt_sell_prices_day))[0]

    best = {
        "execute": False,
        "charge_indices": [],
        "discharge_indices": [],
        "avg_charge_price": np.nan,
        "avg_discharge_price": np.nan,
        "expected_net_spread_eur_per_mwh": np.nan,
        "reason": "Aucune combinaison aFRR meilleure que wholesale.",
    }

    if len(charge_candidate) < n_qh or len(discharge_candidate) < n_qh:
        return best

    eligible_idx = np.where(eligible)[0]
    for split_abs in eligible_idx:
        charge_pool = charge_candidate[charge_candidate < split_abs]
        discharge_pool = discharge_candidate[discharge_candidate > split_abs]
        if len(charge_pool) < n_qh or len(discharge_pool) < n_qh:
            continue

        selected_charge = np.sort(charge_pool[np.argsort(charge_prices_day[charge_pool])[:n_qh]])
        selected_discharge = np.sort(discharge_pool[np.argsort(-discharge_prices_day[discharge_pool])[:n_qh]])
        if selected_charge.max() >= selected_discharge.min():
            continue

        avg_charge_price = float(np.mean(charge_prices_day[selected_charge]))
        avg_discharge_price = float(np.mean(discharge_prices_day[selected_discharge]))
        net_spread = (
            avg_discharge_price
            - avg_charge_price / max(eta_charge * eta_discharge, 1e-12)
            - afrr_cycle_cost_eur_per_mwh
        )

        if (not np.isfinite(best["expected_net_spread_eur_per_mwh"])) or net_spread > best["expected_net_spread_eur_per_mwh"]:
            best = {
                "execute": net_spread >= afrr_min_spread_eur_per_mwh,
                "charge_indices": selected_charge.tolist(),
                "discharge_indices": selected_discharge.tolist(),
                "avg_charge_price": avg_charge_price,
                "avg_discharge_price": avg_discharge_price,
                "expected_net_spread_eur_per_mwh": float(net_spread),
                "reason": "OK" if net_spread >= afrr_min_spread_eur_per_mwh else "Spread insuffisant.",
            }

    if not best["execute"]:
        best["charge_indices"] = []
        best["discharge_indices"] = []
    return best

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


def simulate_afrr_night_arbitrage(inputs: SimulationInputs, result_hourly: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not inputs.enable_afrr:
        return {
            "afrr_charge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_discharge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_soc_qh": np.asarray(result_hourly["soc"][:-1], dtype=float),
            "afrr_charge_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_sale_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_cycle_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_net_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_energy_down_activated_qh": np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_energy_up_activated_qh": np.zeros(QH_PER_YEAR, dtype=int),
            "selected_charge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
            "selected_discharge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_energy_eligible_qh": np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_daily_log": pd.DataFrame(),
        }

    if inputs.afrr_charge_price_qh is None or inputs.afrr_discharge_price_qh is None:
        raise ValueError("Les courbes de prix aFRR quart-horaires doivent être fournies si aFRR est activé.")

    charge_prices_qh = _validate_array_length(inputs.afrr_charge_price_qh, "Prix aFRR charge", QH_PER_YEAR)
    discharge_prices_qh = _validate_array_length(inputs.afrr_discharge_price_qh, "Prix aFRR décharge", QH_PER_YEAR)
    grid_buy_price_qh = _validate_array_length(inputs.grid_buy_price, "Prix achat réseau BESS", QH_PER_YEAR)
    batt_sell_price_qh = _validate_array_length(inputs.batt_sell_price, "Prix vente BESS", QH_PER_YEAR)

    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)

    # aFRR Energy eligibility window. Overnight windows are supported, e.g. 20 -> 8.
    eligible_mask_qh = _hour_window_mask(inputs.afrr_night_start_hour, inputs.afrr_night_end_hour, QH_PER_YEAR)

    afrr_charge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_discharge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_charge_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_sale_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_cycle_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_net_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_soc_qh = np.zeros(QH_PER_YEAR, dtype=float)
    down_activated_qh = np.zeros(QH_PER_YEAR, dtype=int)
    up_activated_qh = np.zeros(QH_PER_YEAR, dtype=int)
    selected_charge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_discharge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    up_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)
    down_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    soc_current = min(max(float(result_hourly["soc"][0]), min_soc_mwh), max_soc_mwh)
    max_charge_input_qh = inputs.batt_power_mw * QH_DT_HOURS
    max_discharge_output_qh = inputs.batt_power_mw * QH_DT_HOURS
    max_export_qh = inputs.grid_export_limit_mw * QH_DT_HOURS

    daily_logs = []

    if inputs.enable_afrr_capacity:
        if inputs.afrr_capacity_selected_market_h is None:
            capacity_selected_h = np.full(QH_PER_YEAR, "none", dtype=object)
        else:
            capacity_selected_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
            if len(capacity_selected_h) != QH_PER_YEAR:
                raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")

        # Capacity-linked aFRR energy is represented as expected activated MWh on every awarded interval.
        # Activation percentages scale MWh, not the number of selected intervals.
        # IMPORTANT FIX: the physical aFRR energy dispatch now follows the same
        # expected activated MWh arrays used in simulate_afrr_capacity() for the
        # market-value comparison. This keeps the selected aFRR capacity value and
        # the actual aFRR energy dispatch consistent, subject only to hard physical
        # SOC, power and export constraints.
        down_selected_qh = (capacity_selected_h == "down")
        up_selected_qh = (capacity_selected_h == "up")
        down_selected_h = down_selected_qh.astype(int)
        up_selected_h = up_selected_qh.astype(int)

        if inputs.afrr_expected_down_activated_mwh_qh is not None:
            expected_down_dispatch_qh = _validate_array_length(
                inputs.afrr_expected_down_activated_mwh_qh,
                "Expected aFRR Down activated MWh",
                QH_PER_YEAR,
            )
        else:
            activation_down_factor = min(max(float(inputs.afrr_energy_down_activation_pct) / 100.0, 0.0), 1.0)
            expected_down_dispatch_qh = down_selected_qh.astype(float) * max_charge_input_qh * activation_down_factor

        if inputs.afrr_expected_up_activated_mwh_qh is not None:
            expected_up_dispatch_qh = _validate_array_length(
                inputs.afrr_expected_up_activated_mwh_qh,
                "Expected aFRR Up activated MWh",
                QH_PER_YEAR,
            )
        else:
            activation_up_factor = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
            expected_up_dispatch_qh = up_selected_qh.astype(float) * max_discharge_output_qh * activation_up_factor

        # Ensure non-awarded directions cannot create activation.
        expected_down_dispatch_qh = np.where(down_selected_qh, np.maximum(expected_down_dispatch_qh, 0.0), 0.0)
        expected_up_dispatch_qh = np.where(up_selected_qh, np.maximum(expected_up_dispatch_qh, 0.0), 0.0)

        up_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)
        down_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)

        for t in range(QH_PER_YEAR):
            if not eligible_mask_qh[t]:
                afrr_soc_qh[t] = soc_current
                continue

            if down_selected_qh[t]:
                target_input_qh = min(float(expected_down_dispatch_qh[t]), max_charge_input_qh)
                feasible_input_qh = min(
                    target_input_qh,
                    max(max_soc_mwh - soc_current, 0.0) / max(inputs.eta_charge, 1e-12),
                )
                down_activation_shortfall_qh[t] = max(target_input_qh - feasible_input_qh, 0.0)
                if feasible_input_qh > 1e-12:
                    afrr_charge_qh_mwh[t] = feasible_input_qh
                    afrr_charge_cost_qh_eur[t] = feasible_input_qh * charge_prices_qh[t]
                    afrr_net_revenue_qh_eur[t] -= afrr_charge_cost_qh_eur[t]
                    soc_current += feasible_input_qh * inputs.eta_charge
                    down_activated_qh[t] = 1
                    selected_charge_market_qh[t] = "afrr"
            elif up_selected_qh[t]:
                pv_direct_t = float(np.asarray(result_hourly.get("pv_direct", np.zeros(QH_PER_YEAR)), dtype=float)[t])
                export_room_t = max(max_export_qh - pv_direct_t, 0.0)
                target_discharge_qh = min(float(expected_up_dispatch_qh[t]), max_discharge_output_qh)
                feasible_discharge_qh = min(
                    target_discharge_qh,
                    export_room_t,
                    max(soc_current - min_soc_mwh, 0.0) * inputs.eta_discharge,
                )
                up_activation_shortfall_qh[t] = max(target_discharge_qh - feasible_discharge_qh, 0.0)
                if feasible_discharge_qh > 1e-12:
                    soc_removed = feasible_discharge_qh / max(inputs.eta_discharge, 1e-12)
                    theoretical_cycle_cost = soc_removed * inputs.afrr_cycle_cost_eur_per_mwh
                    expected_sale_revenue = feasible_discharge_qh * discharge_prices_qh[t]
                    # Do not re-apply the cycle-cost hurdle here: the expected aFRR
                    # capacity selection already included degradation/cycle cost in
                    # the value comparison. Re-applying it would make actual dispatch
                    # diverge from the expected MWh used for market selection.
                    afrr_discharge_qh_mwh[t] = feasible_discharge_qh
                    afrr_sale_revenue_qh_eur[t] = expected_sale_revenue
                    afrr_cycle_cost_qh_eur[t] = theoretical_cycle_cost
                    afrr_net_revenue_qh_eur[t] += afrr_sale_revenue_qh_eur[t]
                    soc_current -= soc_removed
                    up_activated_qh[t] = 1
                    selected_discharge_market_qh[t] = "afrr"

            soc_current = min(max(soc_current, min_soc_mwh), max_soc_mwh)
            afrr_soc_qh[t] = soc_current

        daily_logs.append({
            "day": pd.NaT,
            "mode": "capacity_activated",
            "executed": bool(down_activated_qh.any() or up_activated_qh.any()),
            "down_activation_pct": inputs.afrr_energy_down_activation_pct,
            "up_activation_pct": inputs.afrr_energy_up_activation_pct,
            "down_awarded_hours": int(np.sum(capacity_selected_h == "down")),
            "up_awarded_hours": int(np.sum(capacity_selected_h == "up")),
            "down_activated_hours": int(np.sum(down_selected_h)),
            "up_activated_hours": int(np.sum(up_selected_h)),
            "charge_cost_eur": float(afrr_charge_cost_qh_eur.sum()),
            "sale_revenue_eur": float(afrr_sale_revenue_qh_eur.sum()),
            "cycle_cost_eur": float(afrr_cycle_cost_qh_eur.sum()),
            "net_revenue_eur": float(afrr_net_revenue_qh_eur.sum()),
            "reason": "Capacity directional activation; aFRR cycle cost used as reference hurdle on upward activation, not deducted from cash revenue.",
        })

    else:
        df = pd.DataFrame({
            "datetime": idx_qh,
            "charge_price": charge_prices_qh,
            "discharge_price": discharge_prices_qh,
            "grid_buy_price": grid_buy_price_qh,
            "batt_sell_price": batt_sell_price_qh,
            "eligible": eligible_mask_qh,
        })
        df["day"] = df["datetime"].dt.date

        for day, group in df.groupby("day", sort=True):
            group_idx = group.index.to_numpy()
            best_trade = _select_best_daily_afrr_competing_blocks(
                charge_prices_day=group["charge_price"].to_numpy(dtype=float),
                discharge_prices_day=group["discharge_price"].to_numpy(dtype=float),
                grid_buy_prices_day=group["grid_buy_price"].to_numpy(dtype=float),
                batt_sell_prices_day=group["batt_sell_price"].to_numpy(dtype=float),
                eligible_mask_day=group["eligible"].to_numpy(dtype=bool),
                eta_charge=inputs.eta_charge,
                eta_discharge=inputs.eta_discharge,
                afrr_cycle_cost_eur_per_mwh=inputs.afrr_cycle_cost_eur_per_mwh,
                afrr_min_spread_eur_per_mwh=inputs.afrr_min_spread_eur_per_mwh,
                n_qh=int(inputs.afrr_n_qh_per_side),
            )

            selected_charge_abs_idx = []
            selected_discharge_abs_idx = []
            charged_input_mwh_total = 0.0
            discharged_mwh_total = 0.0
            charge_cost_eur_total = 0.0
            sale_revenue_eur_total = 0.0
            cycle_cost_eur_total = 0.0

            if best_trade["execute"]:
                for rel_idx in [int(i) for i in best_trade["charge_indices"]]:
                    t = int(group_idx[rel_idx])
                    input_this_qh = min(max_charge_input_qh, max(max_soc_mwh - soc_current, 0.0) / max(inputs.eta_charge, 1e-12))
                    if input_this_qh <= 1e-12:
                        continue
                    afrr_charge_qh_mwh[t] = input_this_qh
                    afrr_charge_cost_qh_eur[t] = input_this_qh * charge_prices_qh[t]
                    afrr_net_revenue_qh_eur[t] -= afrr_charge_cost_qh_eur[t]
                    soc_current += input_this_qh * inputs.eta_charge
                    down_activated_qh[t] = 1
                    selected_charge_market_qh[t] = "afrr"
                    selected_charge_abs_idx.append(t)
                    charged_input_mwh_total += input_this_qh
                    charge_cost_eur_total += afrr_charge_cost_qh_eur[t]
                    afrr_soc_qh[t] = soc_current

                for rel_idx in [int(i) for i in best_trade["discharge_indices"]]:
                    t = int(group_idx[rel_idx])
                    pv_direct_t = float(np.asarray(result_hourly.get("pv_direct", np.zeros(QH_PER_YEAR)), dtype=float)[t])
                    export_room_t = max(max_export_qh - pv_direct_t, 0.0)
                    discharge_this_qh = min(
                        max_discharge_output_qh,
                        export_room_t,
                        max(soc_current - min_soc_mwh, 0.0) * inputs.eta_discharge,
                    )
                    if discharge_this_qh <= 1e-12:
                        continue
                    soc_removed = discharge_this_qh / max(inputs.eta_discharge, 1e-12)
                    theoretical_cycle_cost = soc_removed * inputs.afrr_cycle_cost_eur_per_mwh
                    expected_sale_revenue = discharge_this_qh * discharge_prices_qh[t]
                    # Reference-only aFRR cycle-cost hurdle: skip activation if the
                    # activation value does not cover the theoretical degradation cost.
                    # The cost is NOT deducted from reported cash revenue when accepted.
                    if expected_sale_revenue <= theoretical_cycle_cost + 1e-12:
                        continue
                    afrr_discharge_qh_mwh[t] = discharge_this_qh
                    afrr_sale_revenue_qh_eur[t] = expected_sale_revenue
                    afrr_cycle_cost_qh_eur[t] = theoretical_cycle_cost
                    afrr_net_revenue_qh_eur[t] += afrr_sale_revenue_qh_eur[t]  # aFRR cycle cost is a decision/reference metric only, not deducted from cash revenue
                    soc_current -= soc_removed
                    up_activated_qh[t] = 1
                    selected_discharge_market_qh[t] = "afrr"
                    selected_discharge_abs_idx.append(t)
                    discharged_mwh_total += discharge_this_qh
                    sale_revenue_eur_total += afrr_sale_revenue_qh_eur[t]
                    cycle_cost_eur_total += afrr_cycle_cost_qh_eur[t]
                    afrr_soc_qh[t] = soc_current

            group_soc_missing = afrr_soc_qh[group_idx] == 0.0
            afrr_soc_qh[group_idx[group_soc_missing]] = soc_current
            daily_logs.append({
                "day": pd.to_datetime(day),
                "mode": "merchant_competing_routes",
                "executed": bool(len(selected_charge_abs_idx) or len(selected_discharge_abs_idx)),
                "charge_qh_indices": selected_charge_abs_idx,
                "discharge_qh_indices": selected_discharge_abs_idx,
                "charge_times": [idx_qh[i] for i in selected_charge_abs_idx],
                "discharge_times": [idx_qh[i] for i in selected_discharge_abs_idx],
                "avg_charge_price_eur_per_mwh": best_trade.get("avg_charge_price", np.nan),
                "avg_discharge_price_eur_per_mwh": best_trade.get("avg_discharge_price", np.nan),
                "expected_net_spread_eur_per_mwh": best_trade.get("expected_net_spread_eur_per_mwh", np.nan),
                "charged_input_mwh": charged_input_mwh_total,
                "discharged_mwh": discharged_mwh_total,
                "charge_cost_eur": charge_cost_eur_total,
                "sale_revenue_eur": sale_revenue_eur_total,
                "cycle_cost_eur": cycle_cost_eur_total,
                "net_revenue_eur": sale_revenue_eur_total - charge_cost_eur_total,  # aFRR cycle cost reference-only, not deducted
                "reason": best_trade.get("reason", "OK"),
            })

    return {
        "afrr_charge_qh_mwh": afrr_charge_qh_mwh,
        "afrr_discharge_qh_mwh": afrr_discharge_qh_mwh,
        "afrr_soc_qh": afrr_soc_qh,
        "afrr_charge_cost_qh_eur": afrr_charge_cost_qh_eur,
        "afrr_sale_revenue_qh_eur": afrr_sale_revenue_qh_eur,
        "afrr_cycle_cost_qh_eur": afrr_cycle_cost_qh_eur,
        "afrr_net_revenue_qh_eur": afrr_net_revenue_qh_eur,
        "afrr_energy_down_activated_qh": down_activated_qh,
        "afrr_energy_up_activated_qh": up_activated_qh,
        "afrr_up_activation_shortfall_qh_mwh": up_activation_shortfall_qh,
        "afrr_down_activation_shortfall_qh_mwh": down_activation_shortfall_qh,
        "selected_charge_market_qh": selected_charge_market_qh,
        "selected_discharge_market_qh": selected_discharge_market_qh,
        "afrr_energy_eligible_qh": eligible_mask_qh.astype(int),
        "afrr_daily_log": pd.DataFrame(daily_logs),
    }
