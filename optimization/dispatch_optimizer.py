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
from optimization.forward_curves import compute_forward_cross_market_value_curves

def optimize_dispatch_dp(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    pv_sellable = _validate_array_length(inputs.solar_profile, "La production PV nette 15 minutes sellable")
    pv_sellable = np.maximum(pv_sellable, 0.0)

    if inputs.curtailed_pv_recoverable_mwh is None:
        pv_recoverable = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        pv_recoverable = _validate_array_length(inputs.curtailed_pv_recoverable_mwh, "PV curtailed recoverable")
        pv_recoverable = np.maximum(pv_recoverable, 0.0)

    pv_price = _validate_array_length(inputs.pv_price, "Le prix PV")
    batt_sell = _validate_array_length(inputs.batt_sell_price, "Le prix de vente batterie")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Le prix d'achat réseau")

    idx = build_quarter_hour_index(DEFAULT_YEAR)

    df_thresholds = pd.DataFrame({
        "datetime": idx,
        "grid_buy": grid_buy,
        "batt_sell": batt_sell,
    })
    df_thresholds["day"] = df_thresholds["datetime"].dt.date

    charge_threshold_series = df_thresholds.groupby("day")["grid_buy"].transform(
        lambda x: np.percentile(x, inputs.charge_quantile)
    ).to_numpy()

    discharge_threshold_series = df_thresholds.groupby("day")["batt_sell"].transform(
        lambda x: np.percentile(x, inputs.discharge_quantile)
    ).to_numpy()

    if np.any(~np.isfinite(pv_sellable)) or np.any(~np.isfinite(pv_price)) or np.any(~np.isfinite(batt_sell)) or np.any(~np.isfinite(grid_buy)):
        raise ValueError("Une ou plusieurs séries contiennent des valeurs invalides.")
    if inputs.batt_power_mw < 0 or inputs.batt_energy_mwh < 0:
        raise ValueError("La puissance et la capacité batterie doivent être positives.")
    if inputs.eta_charge <= 0 or inputs.eta_charge > 1:
        raise ValueError("Le rendement de charge doit être compris entre 0 et 1.")
    if inputs.eta_discharge <= 0 or inputs.eta_discharge > 1:
        raise ValueError("Le rendement de décharge doit être compris entre 0 et 1.")
    if inputs.initial_soc_mwh < 0 or inputs.final_soc_mwh < 0:
        raise ValueError("Les SOC initial et final doivent être positifs.")
    if inputs.initial_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC initial ne peut pas dépasser la capacité batterie.")
    if inputs.final_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC final ne peut pas dépasser la capacité batterie.")
    if not (0.0 <= inputs.min_soc_pct <= 100.0):
        raise ValueError("Minimum SOC batterie (%) doit être compris entre 0 et 100 %.")
    if not (0.0 <= inputs.max_soc_pct <= 100.0):
        raise ValueError("Maximum SOC batterie (%) doit être compris entre 0 et 100 %.")
    if inputs.min_soc_pct >= inputs.max_soc_pct:
        raise ValueError("Minimum SOC batterie (%) doit être strictement inférieur au Maximum SOC batterie (%).")

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0

    if inputs.initial_soc_mwh < min_soc_mwh or inputs.initial_soc_mwh > max_soc_mwh:
        raise ValueError("Le SOC initial doit être compris dans la plage SOC autorisée.")
    if inputs.final_soc_mwh < min_soc_mwh or inputs.final_soc_mwh > max_soc_mwh:
        raise ValueError("Le SOC final doit être compris dans la plage SOC autorisée.")

    T = len(pv_sellable)
    if T != QH_PER_YEAR:
        raise ValueError("Toutes les séries doivent contenir 35040 pas de 15 minutes.")

    if inputs.max_cycles_per_year < 0:
        raise ValueError("Cycles max / an doit être positif ou nul.")

    max_annual_discharge_mwh = float(inputs.max_cycles_per_year) * float(inputs.batt_energy_mwh)
    minimum_discharge_to_reach_final_mwh = max(inputs.initial_soc_mwh - inputs.final_soc_mwh, 0.0) * inputs.eta_discharge
    if max_annual_discharge_mwh + 1e-9 < minimum_discharge_to_reach_final_mwh:
        raise ValueError(
            "Cycles max / an est trop faible pour atteindre le SOC final demandé. "
            f"Minimum requis: {minimum_discharge_to_reach_final_mwh / max(inputs.batt_energy_mwh, 1e-12):.3f} cycles/an."
        )

    # aFRR Capacity awarded hours reserve the battery and block wholesale battery actions.
    if inputs.afrr_capacity_selected_market_h is None:
        afrr_capacity_selected_market_h = np.full(T, "none", dtype=object)
    else:
        afrr_capacity_selected_market_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
        if len(afrr_capacity_selected_market_h) != T:
            raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")

    battery_blocked_by_afrr_capacity = np.isin(afrr_capacity_selected_market_h, ["up", "down"])

    soc_steps = int(max(21, inputs.soc_steps))
    soc_grid = np.linspace(min_soc_mwh, max_soc_mwh, soc_steps)

    def nearest_state_index(value: float) -> int:
        value = min(max(value, min_soc_mwh), max_soc_mwh)
        return int(np.argmin(np.abs(soc_grid - value)))

    init_idx = nearest_state_index(inputs.initial_soc_mwh)
    final_idx = nearest_state_index(inputs.final_soc_mwh)

    DT = QH_DT_HOURS
    charge_soc_max = inputs.batt_power_mw * inputs.eta_charge * DT
    discharge_soc_max = inputs.batt_power_mw * DT / inputs.eta_discharge

    transitions = []
    for i, soc in enumerate(soc_grid):
        j_min = np.searchsorted(soc_grid, max(min_soc_mwh, soc - discharge_soc_max), side="left")
        j_max = np.searchsorted(soc_grid, min(max_soc_mwh, soc + charge_soc_max), side="right") - 1
        transitions.append(np.arange(j_min, j_max + 1, dtype=int))

    forward_curves_dp = compute_forward_cross_market_value_curves(inputs)
    future_best_sell_price_from_t = np.maximum(
        forward_curves_dp["future_expected_wholesale_value_eur_per_mwh"],
        forward_curves_dp["future_expected_afrr_up_value_eur_per_mwh"],
    )
    future_best_market_type_from_t = forward_curves_dp["future_best_market_type"]
    future_best_sell_price_from_t = np.nan_to_num(future_best_sell_price_from_t, nan=-1e30, posinf=1e30, neginf=-1e30)
    
    def run_dp_once(
        required_discharge_price_estimate: np.ndarray,
        annual_cycle_budget_penalty_eur_per_mwh: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        neg_inf = -1e30
        value_next = np.full(soc_steps, neg_inf, dtype=float)
        value_next[final_idx] = 0.0
        policy_next = np.full((T, soc_steps), -1, dtype=np.int16 if soc_steps < 32000 else np.int32)

        estimate_gate = np.asarray(required_discharge_price_estimate, dtype=float).reshape(-1)
        if len(estimate_gate) != T:
            raise ValueError("La courbe estimée de prix requis de décharge a une mauvaise longueur.")
        estimate_gate = np.nan_to_num(estimate_gate, nan=-1e30, posinf=1e30, neginf=-1e30)

        for t in range(T - 1, -1, -1):
            value_now = np.full(soc_steps, neg_inf, dtype=float)
            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]
            pv_price_t = pv_price[t]
            batt_sell_t = batt_sell[t]
            grid_buy_t = grid_buy[t]

            for i in range(soc_steps):
                best_val = neg_inf
                best_j = -1
                soc_i = soc_grid[i]

                for j in transitions[i]:
                    delta_soc = soc_grid[j] - soc_i

                    # If aFRR Capacity is awarded for this hour, the battery must be reserved:
                    # no PV-to-battery, curtailed-PV-to-battery, grid charge or wholesale discharge.
                    if battery_blocked_by_afrr_capacity[t] and abs(delta_soc) > 1e-12:
                        continue

                    pv_direct_candidate = pv_sellable_t
                    sellable_pv_to_batt = 0.0
                    recoverable_pv_to_batt = 0.0
                    grid_charge = 0.0
                    discharge_candidate = 0.0
                    cycle_penalty = 0.0

                    if delta_soc > 1e-12:
                        charge_input = delta_soc / inputs.eta_charge

                        recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                        remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                        sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                        remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                        grid_charge = max(remaining_after_sellable, 0.0)
                        pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt
                        pv_is_producing = (pv_sellable_t + pv_recoverable_t) > 1e-9
                        
                        # Block grid charging when PV is producing
                        if grid_charge > 1e-9 and pv_sellable_t > 1e-9:
                            continue

                        if grid_charge > 1e-9 and pv_is_producing:
                            continue
                            
                        if grid_charge > 1e-9:
                            if grid_buy_t > charge_threshold_series[t]:
                                continue
                        
                            future_best_sell_price = future_best_sell_price_from_t[t]
                            future_route = future_best_market_type_from_t[t]
                            if future_route == "afrr_up":
                                required_spread = max(inputs.afrr_up_cross_market_min_spread_eur_per_mwh, inputs.afrr_min_spread_eur_per_mwh)
                            else:
                                required_spread = inputs.min_spread_arbitrage_eur_per_mwh
                        
                            required_future_sell_price = (
                                grid_buy_t / max(inputs.eta_charge * inputs.eta_discharge, 1e-12)
                                + required_spread
                                + inputs.cycle_cost_eur_per_mwh
                            )
                        
                            if future_best_sell_price < required_future_sell_price:
                                continue

                    elif delta_soc < -1e-12:
                        # PV priority rule with export headroom:
                        # PV keeps priority on the grid export limit, but BESS may
                        # discharge during PV production if PV does not already fill
                        # the available injection capacity.
                        pv_export_headroom = max(inputs.grid_export_limit_mw * QH_DT_HOURS - pv_sellable_t, 0.0)
                        if pv_export_headroom <= 1e-9:
                            continue

                        discharge_candidate = min((-delta_soc) * inputs.eta_discharge, pv_export_headroom)

                        if discharge_candidate > 1e-9:
                            if batt_sell_t < discharge_threshold_series[t]:
                                continue
                            if batt_sell_t < estimate_gate[t]:
                                continue

                    total_export = pv_direct_candidate + discharge_candidate

                    if total_export > inputs.grid_export_limit_mw * QH_DT_HOURS:
                        excess = total_export - inputs.grid_export_limit_mw * QH_DT_HOURS
                        reduction_pv = min(excess, pv_direct_candidate)
                        pv_direct_candidate -= reduction_pv
                        excess -= reduction_pv

                        if excess > 0:
                            discharge_candidate = max(discharge_candidate - excess, 0.0)

                        # Cycle cost is applied below for every MWh actually discharged,
                        # not only when the grid export limit is binding.

                    if discharge_candidate > 1e-12:
                        # Marginal degradation / wear cost, in EUR per MWh discharged.
                        # This makes cycle cost economically effective in the dispatch decision.
                        cycle_penalty = discharge_candidate * inputs.cycle_cost_eur_per_mwh

                    reward = pv_direct_candidate * pv_price_t

                    if delta_soc > 1e-12:
                        reward -= grid_charge * grid_buy_t
                    elif delta_soc < -1e-12:
                        reward += discharge_candidate * batt_sell_t
                        reward -= cycle_penalty
                        # Shadow price used only by the optimizer to allocate a limited
                        # annual cycle budget to the best spreads over the full year.
                        reward -= annual_cycle_budget_penalty_eur_per_mwh * discharge_candidate

                    total_val = reward + value_next[j]
                    if total_val > best_val:
                        best_val = total_val
                        best_j = int(j)

                value_now[i] = best_val
                policy_next[t, i] = best_j

            value_next = value_now

        if np.all(value_next == neg_inf):
            raise RuntimeError("DP failed: all states unreachable")

        soc = np.zeros(T + 1, dtype=float)
        soc[0] = soc_grid[init_idx]
        state = init_idx

        pv_direct = np.zeros(T, dtype=float)
        pv_to_batt = np.zeros(T, dtype=float)
        pv_curtailed_to_battery = np.zeros(T, dtype=float)
        grid_charge = np.zeros(T, dtype=float)
        discharge = np.zeros(T, dtype=float)
        batt_sale_revenue = np.zeros(T, dtype=float)
        grid_charge_cost = np.zeros(T, dtype=float)
        wholesale_cycle_cost = np.zeros(T, dtype=float)
        pv_direct_revenue = np.zeros(T, dtype=float)
        avg_stored_charge_price = np.full(T + 1, np.nan, dtype=float)
        required_discharge_price = np.full(T, np.nan, dtype=float)
        stored_energy_value_eur = 0.0
        stored_energy_mwh = soc[0]

        avg_stored_charge_price[0] = 0.0 if stored_energy_mwh > 1e-9 else np.nan

        for t in range(T):
            next_state = int(policy_next[t, state])
            if next_state < 0:
                raise RuntimeError(f"Policy failure at t={t}, state={state}")

            delta_soc = soc_grid[next_state] - soc_grid[state]
            if battery_blocked_by_afrr_capacity[t] and abs(delta_soc) > 1e-9:
                raise RuntimeError(f"aFRR Capacity wholesale block violated at t={t}")
            soc[t + 1] = soc_grid[next_state]

            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]

            pv_direct_candidate = pv_sellable_t
            sellable_pv_to_batt = 0.0
            recoverable_pv_to_batt = 0.0
            grid_charge[t] = 0.0
            discharge[t] = 0.0

            if delta_soc > 1e-12:
                charge_input = delta_soc / inputs.eta_charge

                recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                grid_charge[t] = max(remaining_after_sellable, 0.0)
                pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt

            elif delta_soc < -1e-12:
                # Safety mirror of the DP rule above: allow discharge only into
                # remaining grid export headroom after PV priority.
                pv_export_headroom = max(inputs.grid_export_limit_mw * QH_DT_HOURS - pv_sellable_t, 0.0)
                discharge[t] = min((-delta_soc) * inputs.eta_discharge, pv_export_headroom)

            pv_to_batt[t] = sellable_pv_to_batt
            pv_curtailed_to_battery[t] = recoverable_pv_to_batt

            if delta_soc > 1e-12:
                charge_cost_eur = (
                    sellable_pv_to_batt * pv_price[t] +
                    grid_charge[t] * grid_buy[t]
                    # recoverable_pv_to_batt enters at zero opportunity cost
                )
                stored_energy_value_eur += charge_cost_eur
                stored_energy_mwh += delta_soc

            elif delta_soc < -1e-12:
                avg_cost_now = stored_energy_value_eur / max(stored_energy_mwh, 1e-9)
                energy_removed_from_soc = -delta_soc
                cost_removed_eur = avg_cost_now * energy_removed_from_soc
                stored_energy_value_eur = max(stored_energy_value_eur - cost_removed_eur, 0.0)
                stored_energy_mwh = max(stored_energy_mwh - energy_removed_from_soc, 0.0)

            if stored_energy_mwh > 1e-9:
                avg_stored_charge_price[t + 1] = stored_energy_value_eur / stored_energy_mwh
            else:
                avg_stored_charge_price[t + 1] = np.nan

            if np.isfinite(avg_stored_charge_price[t]):
                required_discharge_price[t] = (
                    avg_stored_charge_price[t]
                    + inputs.min_spread_arbitrage_eur_per_mwh
                    + inputs.cycle_cost_eur_per_mwh
                )

            total_export = pv_direct_candidate + discharge[t]
            if total_export > inputs.grid_export_limit_mw:
                excess = total_export - inputs.grid_export_limit_mw
                reduction_pv = min(excess, pv_direct_candidate)
                pv_direct_candidate -= reduction_pv
                excess -= reduction_pv
                if excess > 0:
                    discharge[t] = max(discharge[t] - excess, 0.0)

            pv_direct[t] = max(pv_direct_candidate, 0.0)
            pv_direct_revenue[t] = pv_direct[t] * pv_price[t]
            batt_sale_revenue[t] = discharge[t] * batt_sell[t]
            grid_charge_cost[t] = grid_charge[t] * grid_buy[t]
            # Option A: cycle cost is a dispatch hurdle and an informational theoretical degradation metric only.
            # It is NOT deducted from cash revenue.
            wholesale_cycle_cost[t] = discharge[t] * inputs.cycle_cost_eur_per_mwh
            state = next_state

        total_direct_pv_revenue = float(pv_direct_revenue.sum())
        total_batt_sale_revenue = float(batt_sale_revenue.sum())
        total_grid_charge_cost = float(grid_charge_cost.sum())
        total_wholesale_cycle_cost = float(wholesale_cycle_cost.sum())
        nightly_revenue_total = float(inputs.nightly_bess_revenue_eur * (T // 24))
        total_revenue = total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
        total_discharged_mwh = float(discharge.sum())
        equivalent_cycles = total_discharged_mwh / max(inputs.batt_energy_mwh, 1e-12)
        remaining_cycle_budget_mwh = max(max_annual_discharge_mwh - total_discharged_mwh, 0.0)

        return {
            "soc": soc,
            "pv_direct": pv_direct,
            "pv_to_batt": pv_to_batt,
            "pv_curtailed_to_battery": pv_curtailed_to_battery,
            "grid_charge": grid_charge,
            "discharge": discharge,
            "pv_direct_revenue": pv_direct_revenue,
            "batt_sale_revenue": batt_sale_revenue,
            "grid_charge_cost": grid_charge_cost,
            "wholesale_cycle_cost_eur": wholesale_cycle_cost,
            "total_direct_pv_revenue": np.array([total_direct_pv_revenue]),
            "total_batt_sale_revenue": np.array([total_batt_sale_revenue]),
            "total_grid_charge_cost": np.array([total_grid_charge_cost]),
            "total_wholesale_cycle_cost_eur": np.array([total_wholesale_cycle_cost]),
            "gross_bess_revenue_before_cycle_cost_eur": np.array([total_batt_sale_revenue]),
            "net_bess_revenue_after_cycle_cost_eur": np.array([total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total]),
            "bess_cash_revenue_eur": np.array([total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total]),
            "nightly_revenue_total": np.array([nightly_revenue_total]),
            "total_revenue": np.array([total_revenue]),
            "equivalent_cycles": np.array([equivalent_cycles]),
            "energy_sold_total_mwh": np.array([pv_direct.sum() + total_discharged_mwh]),
            "energy_shifted_mwh": np.array([total_discharged_mwh]),
            "max_cycles_per_year": np.array([float(inputs.max_cycles_per_year)]),
            "annual_discharge_cap_mwh": np.array([max_annual_discharge_mwh]),
            "remaining_cycle_budget_mwh": np.array([remaining_cycle_budget_mwh]),
            "annual_cycle_budget_penalty_eur_per_mwh": np.array([float(annual_cycle_budget_penalty_eur_per_mwh)]),
            "pv_direct_sold_mwh": np.array([pv_direct.sum()]),
            "avg_stored_charge_price": avg_stored_charge_price,
            "required_discharge_price": required_discharge_price,
            "hourly_datetime": idx,
            "required_discharge_price_gate_estimate": estimate_gate,
            "afrr_capacity_selected_market_h": afrr_capacity_selected_market_h,
            "battery_blocked_by_afrr_capacity": battery_blocked_by_afrr_capacity.astype(int),
        }

    def run_dp_with_annual_cycle_cap(required_discharge_price_estimate: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the annual DP with a global annual discharge budget.

        A direct SOC x cycle-budget DP would be very large for 35040 quarter-hour steps.
        This uses the equivalent Lagrangian form: a shadow price is applied to
        every MWh discharged, then found by bisection. Because the DP still sees
        the full 35040-step quarter-hour horizon, it can skip low-value cycles early in the
        year and keep the limited annual cycle budget for better spreads later.
        """
        cap_tolerance_mwh = max(1e-6, 1e-6 * max(inputs.batt_energy_mwh, 1.0))
        uncapped = run_dp_once(required_discharge_price_estimate, 0.0)
        if float(uncapped["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
            return uncapped

        low_penalty = 0.0
        high_penalty = max(1.0, float(np.nanmax(batt_sell) - np.nanmin(grid_buy) + inputs.min_spread_arbitrage_eur_per_mwh))
        capped = run_dp_once(required_discharge_price_estimate, high_penalty)

        # Increase the shadow price until the selected annual dispatch respects the cap.
        for _ in range(3):
            if float(capped["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
                break
            low_penalty = high_penalty
            high_penalty *= 2.0
            capped = run_dp_once(required_discharge_price_estimate, high_penalty)

        if float(capped["energy_shifted_mwh"][0]) > max_annual_discharge_mwh + cap_tolerance_mwh:
            raise RuntimeError(
                "Impossible de respecter Cycles max / an avec les contraintes SOC initial/final et les pas de SOC choisis. "
                "Augmentez Cycles max / an, réduisez le SOC final requis, ou augmentez le nombre de pas de SOC."
            )

        best_capped = capped
        for _ in range(3):
            mid_penalty = 0.5 * (low_penalty + high_penalty)
            candidate = run_dp_once(required_discharge_price_estimate, mid_penalty)
            if float(candidate["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
                high_penalty = mid_penalty
                best_capped = candidate
            else:
                low_penalty = mid_penalty

        return best_capped

    max_passes = 2
    required_estimate = np.full(T, -1e30, dtype=float)
    final_result = None

    for _ in range(max_passes):
        candidate = run_dp_with_annual_cycle_cap(required_estimate)

        new_estimate = np.nan_to_num(
            candidate["required_discharge_price"],
            nan=-1e30,
            posinf=1e30,
            neginf=-1e30,
        )

        # Important: only tighten the gate, never loosen it.
        tightened_estimate = np.maximum(required_estimate, new_estimate)

        discharge_mask = candidate["discharge"] > 1e-6
        valid_required_mask = np.isfinite(candidate["required_discharge_price"])

        violations = (
            discharge_mask
            & valid_required_mask
            & (batt_sell < candidate["required_discharge_price"] - 1e-9)
        )

        final_result = candidate

        if not violations.any() and np.allclose(
            tightened_estimate,
            required_estimate,
            atol=1e-6,
            rtol=0.0,
        ):
            break

        required_estimate = tightened_estimate.copy()

    return final_result
