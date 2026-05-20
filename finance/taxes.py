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

def compute_spain_fee_tax_breakdown(
    gross_inputs: SimulationInputs,
    dispatch_inputs: SimulationInputs,
    result: dict,
    reconciliation: dict | None,
    afrr_capacity_result: dict | None,
    pv_benchmark: dict | None = None,
) -> dict:
    """Reporting-only Spain fee/tax breakdown.

    Gross market prices are used to calculate gross revenues. Variable fees and
    taxes are shown separately. Financial-only taxes are applied after EBITDA and
    never feed back into dispatch.
    """
    rec = reconciliation or {}
    zeros = np.zeros(QH_PER_YEAR, dtype=float)

    pv_direct_mwh = _validate_array_length(result.get("pv_direct", zeros), "PV direct MWh")
    pv_revenue_gross = pv_direct_mwh * _validate_array_length(gross_inputs.pv_price, "gross PV price")

    wholesale_discharge_mwh = _validate_array_length(
        rec.get("wholesale_discharge_qh_mwh", result.get("discharge", zeros)),
        "wholesale discharge MWh",
    )
    wholesale_grid_charge_mwh = _validate_array_length(
        rec.get("wholesale_grid_charge_qh_mwh", result.get("grid_charge", zeros)),
        "wholesale grid charge MWh",
    )

    da_discharge_revenue_gross = wholesale_discharge_mwh * _validate_array_length(gross_inputs.batt_sell_price, "gross DA sell price")
    da_charge_cost_gross = wholesale_grid_charge_mwh * _validate_array_length(gross_inputs.grid_buy_price, "gross DA buy price")

    afrr_charge_mwh = _validate_array_length(rec.get("afrr_charge_qh_mwh", zeros), "aFRR charge MWh")
    afrr_discharge_mwh = _validate_array_length(rec.get("afrr_discharge_qh_mwh", zeros), "aFRR discharge MWh")

    if gross_inputs.afrr_charge_price_qh is not None:
        afrr_charge_cost_gross = afrr_charge_mwh * _validate_array_length(gross_inputs.afrr_charge_price_qh, "gross aFRR DOWN price")
    else:
        afrr_charge_cost_gross = zeros.copy()

    if gross_inputs.afrr_discharge_price_qh is not None:
        afrr_energy_revenue_gross = afrr_discharge_mwh * _validate_array_length(gross_inputs.afrr_discharge_price_qh, "gross aFRR UP price")
    else:
        afrr_energy_revenue_gross = zeros.copy()

    afrr_capacity_revenue_gross = zeros.copy()
    if afrr_capacity_result is not None:
        # Recompute expected gross capacity revenue from gross prices so capacity
        # fees/IVPEE can be shown separately and not double-counted.
        success = min(max(float(gross_inputs.afrr_capacity_success_rate_pct) / 100.0, 0.0), 1.0)
        up_awarded = np.asarray(afrr_capacity_result.get("afrr_capacity_up_awarded_h", zeros), dtype=float).reshape(-1)
        down_awarded = np.asarray(afrr_capacity_result.get("afrr_capacity_down_awarded_h", zeros), dtype=float).reshape(-1)
        if len(up_awarded) != QH_PER_YEAR:
            up_awarded = zeros.copy()
        if len(down_awarded) != QH_PER_YEAR:
            down_awarded = zeros.copy()

        if gross_inputs.afrr_capacity_up_price_h is not None:
            afrr_capacity_revenue_gross += (
                up_awarded
                * _validate_array_length(gross_inputs.afrr_capacity_up_price_h, "gross aFRR UP capacity price")
                * float(gross_inputs.afrr_certified_capacity_up_mw)
                * QH_DT_HOURS
                * success
            )
        if gross_inputs.afrr_capacity_down_price_h is not None:
            afrr_capacity_revenue_gross += (
                down_awarded
                * _validate_array_length(gross_inputs.afrr_capacity_down_price_h, "gross aFRR DOWN capacity price")
                * float(gross_inputs.afrr_certified_capacity_down_mw)
                * QH_DT_HOURS
                * success
            )

    pv_export_fee = pv_direct_mwh * gross_inputs.grid_export_fee_eur_per_mwh
    pv_omie_fee = pv_direct_mwh * gross_inputs.omie_sell_fee_eur_per_mwh
    pv_ree_fee = pv_direct_mwh * gross_inputs.ree_system_fee_eur_per_mwh
    pv_imbalance_cost = pv_direct_mwh * gross_inputs.imbalance_cost_pv_eur_per_mwh

    da_grid_import_fee = wholesale_grid_charge_mwh * gross_inputs.grid_import_fee_eur_per_mwh
    da_buy_omie_fee = wholesale_grid_charge_mwh * gross_inputs.omie_buy_fee_eur_per_mwh
    da_buy_ree_fee = wholesale_grid_charge_mwh * gross_inputs.ree_system_fee_eur_per_mwh
    da_buy_imbalance_cost = wholesale_grid_charge_mwh * gross_inputs.imbalance_cost_bess_eur_per_mwh

    da_export_fee = wholesale_discharge_mwh * gross_inputs.grid_export_fee_eur_per_mwh
    da_sell_omie_fee = wholesale_discharge_mwh * gross_inputs.omie_sell_fee_eur_per_mwh
    da_sell_ree_fee = wholesale_discharge_mwh * gross_inputs.ree_system_fee_eur_per_mwh
    da_sell_imbalance_cost = wholesale_discharge_mwh * gross_inputs.imbalance_cost_bess_eur_per_mwh

    afrr_energy_fee_pct_cost = np.maximum(afrr_energy_revenue_gross, 0.0) * (gross_inputs.afrr_energy_fee_pct / 100.0)
    afrr_energy_fee_variable = (np.abs(afrr_charge_mwh) + np.abs(afrr_discharge_mwh)) * gross_inputs.afrr_energy_fee_eur_per_mwh
    afrr_capacity_fee = np.maximum(afrr_capacity_revenue_gross, 0.0) * (gross_inputs.afrr_capacity_fee_pct / 100.0)

    ivpee_rate = gross_inputs.ivpee_generation_tax_pct / 100.0
    pv_ivpee_tax = np.maximum(pv_revenue_gross, 0.0) * ivpee_rate if gross_inputs.apply_ivpee_to_pv else zeros.copy()
    da_ivpee_tax = np.maximum(da_discharge_revenue_gross, 0.0) * ivpee_rate if gross_inputs.apply_ivpee_to_bess_export else zeros.copy()
    afrr_energy_ivpee_tax = np.maximum(afrr_energy_revenue_gross, 0.0) * ivpee_rate if gross_inputs.apply_ivpee_to_afrr_energy else zeros.copy()
    afrr_capacity_ivpee_tax = np.maximum(afrr_capacity_revenue_gross, 0.0) * ivpee_rate if gross_inputs.apply_ivpee_to_afrr_capacity else zeros.copy()

    total_variable_fees_and_taxes = (
        pv_export_fee + pv_omie_fee + pv_ree_fee + pv_imbalance_cost + pv_ivpee_tax
        + da_grid_import_fee + da_buy_omie_fee + da_buy_ree_fee + da_buy_imbalance_cost
        + da_export_fee + da_sell_omie_fee + da_sell_ree_fee + da_sell_imbalance_cost
        + afrr_energy_fee_pct_cost + afrr_energy_fee_variable + afrr_capacity_fee
        + da_ivpee_tax + afrr_energy_ivpee_tax + afrr_capacity_ivpee_tax
    )

    gross_revenue_before_fees = (
        pv_revenue_gross
        + da_discharge_revenue_gross
        - da_charge_cost_gross
        + afrr_energy_revenue_gross
        - afrr_charge_cost_gross
        + afrr_capacity_revenue_gross
    )
    net_revenue_after_variable_fees = gross_revenue_before_fees - total_variable_fees_and_taxes

    ebitda_before_fixed_costs = float(np.sum(net_revenue_after_variable_fees))
    ebitda_after_fixed_costs = ebitda_before_fixed_costs - float(gross_inputs.local_fixed_tax_eur_per_year)
    taxable_profit = max(ebitda_after_fixed_costs, 0.0)
    corporate_tax = taxable_profit * gross_inputs.corporate_tax_pct / 100.0
    cash_flow_after_corporate_tax = ebitda_after_fixed_costs - corporate_tax
    withholding_tax = max(cash_flow_after_corporate_tax, 0.0) * gross_inputs.withholding_tax_pct / 100.0
    cash_flow_after_tax_and_withholding = cash_flow_after_corporate_tax - withholding_tax

    return {
        "pv_revenue_gross_eur": pv_revenue_gross,
        "da_discharge_revenue_gross_eur": da_discharge_revenue_gross,
        "da_charge_cost_gross_eur": da_charge_cost_gross,
        "afrr_energy_revenue_gross_eur": afrr_energy_revenue_gross,
        "afrr_charge_cost_gross_eur": afrr_charge_cost_gross,
        "afrr_capacity_revenue_gross_eur": afrr_capacity_revenue_gross,
        "pv_export_fee_eur": pv_export_fee,
        "pv_omie_fee_eur": pv_omie_fee,
        "pv_ree_fee_eur": pv_ree_fee,
        "pv_imbalance_cost_eur": pv_imbalance_cost,
        "pv_ivpee_tax_eur": pv_ivpee_tax,
        "da_grid_import_fee_eur": da_grid_import_fee,
        "da_buy_omie_fee_eur": da_buy_omie_fee,
        "da_buy_ree_fee_eur": da_buy_ree_fee,
        "da_buy_imbalance_cost_eur": da_buy_imbalance_cost,
        "da_export_fee_eur": da_export_fee,
        "da_sell_omie_fee_eur": da_sell_omie_fee,
        "da_sell_ree_fee_eur": da_sell_ree_fee,
        "da_sell_imbalance_cost_eur": da_sell_imbalance_cost,
        "da_ivpee_tax_eur": da_ivpee_tax,
        "afrr_energy_fee_pct_cost_eur": afrr_energy_fee_pct_cost,
        "afrr_energy_fee_variable_eur": afrr_energy_fee_variable,
        "afrr_capacity_fee_eur": afrr_capacity_fee,
        "afrr_energy_ivpee_tax_eur": afrr_energy_ivpee_tax,
        "afrr_capacity_ivpee_tax_eur": afrr_capacity_ivpee_tax,
        "total_variable_fees_and_taxes_eur": total_variable_fees_and_taxes,
        "gross_revenue_before_fees_eur": gross_revenue_before_fees,
        "net_revenue_after_variable_fees_eur": net_revenue_after_variable_fees,
        "ebitda_before_fixed_costs_eur": np.array([ebitda_before_fixed_costs]),
        "local_fixed_tax_eur_per_year": np.array([float(gross_inputs.local_fixed_tax_eur_per_year)]),
        "ebitda_after_fixed_costs_eur": np.array([ebitda_after_fixed_costs]),
        "corporate_tax_eur": np.array([corporate_tax]),
        "withholding_tax_eur": np.array([withholding_tax]),
        "cash_flow_after_tax_and_withholding_eur": np.array([cash_flow_after_tax_and_withholding]),
    }

def summarize_spain_fee_tax_breakdown(fee_breakdown: dict) -> pd.DataFrame:
    """One-row-per-metric summary for Streamlit and Excel."""
    def total(key: str) -> float:
        value = fee_breakdown.get(key, np.array([0.0]))
        return float(np.nansum(np.asarray(value, dtype=float)))

    rows = [
        ("Gross revenue before fees", total("gross_revenue_before_fees_eur")),
        ("Total variable fees and taxes", total("total_variable_fees_and_taxes_eur")),
        ("Net revenue after variable fees", total("net_revenue_after_variable_fees_eur")),
        ("Local/fixed taxes", total("local_fixed_tax_eur_per_year")),
        ("EBITDA after fixed costs", total("ebitda_after_fixed_costs_eur")),
        ("Corporate tax", total("corporate_tax_eur")),
        ("Withholding tax", total("withholding_tax_eur")),
        ("Cash flow after tax and withholding", total("cash_flow_after_tax_and_withholding_eur")),
    ]
    return pd.DataFrame([(name, value, "€") for name, value in rows], columns=["Indicateur", "Valeur", "Unité"])
