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
from utils.dataframe_utils import monthly_dataframe

def build_summary_table(
    result: Dict[str, np.ndarray],
    pv_stats: Dict[str, float],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    pv_capture_rate_pct: float,
    bess_capture_rate_pct: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    pv_revenue = float(result["total_direct_pv_revenue"][0])

    wholesale_cycle_cost = float(result["total_wholesale_cycle_cost_eur"][0]) if "total_wholesale_cycle_cost_eur" in result else 0.0
    bess_revenue_base = (
        float(result["total_batt_sale_revenue"][0])
        - float(result["total_grid_charge_cost"][0])
        + float(result["nightly_revenue_total"][0])
    )

    afrr_net_revenue = float(result["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in result else 0.0
    afrr_sale_revenue = float(result["total_afrr_sale_revenue_eur"][0]) if "total_afrr_sale_revenue_eur" in result else 0.0
    afrr_charge_cost = float(result["total_afrr_charge_cost_eur"][0]) if "total_afrr_charge_cost_eur" in result else 0.0
    afrr_cycle_cost = float(result["total_afrr_cycle_cost_eur"][0]) if "total_afrr_cycle_cost_eur" in result else 0.0
    total_cycle_cost = wholesale_cycle_cost + afrr_cycle_cost

    afrr_capacity_up_revenue = float(result["total_afrr_capacity_up_revenue_eur"][0]) if "total_afrr_capacity_up_revenue_eur" in result else 0.0
    afrr_capacity_down_revenue = float(result["total_afrr_capacity_down_revenue_eur"][0]) if "total_afrr_capacity_down_revenue_eur" in result else 0.0
    afrr_capacity_revenue = float(result["total_afrr_capacity_revenue_eur"][0]) if "total_afrr_capacity_revenue_eur" in result else 0.0
    certified_up_mw = float(np.nanmax(result["afrr_certified_capacity_up_mw_h"])) if "afrr_certified_capacity_up_mw_h" in result and len(result["afrr_certified_capacity_up_mw_h"]) else 0.0
    certified_down_mw = float(np.nanmax(result["afrr_certified_capacity_down_mw_h"])) if "afrr_certified_capacity_down_mw_h" in result and len(result["afrr_certified_capacity_down_mw_h"]) else 0.0
    capacity_up_hours = int(np.sum(result["afrr_capacity_up_awarded_h"])) if "afrr_capacity_up_awarded_h" in result else 0
    capacity_down_hours = int(np.sum(result["afrr_capacity_down_awarded_h"])) if "afrr_capacity_down_awarded_h" in result else 0

    bess_revenue_total = bess_revenue_base + afrr_net_revenue + afrr_capacity_revenue
    if "total_revenue_including_afrr_capacity_eur" in result:
        total_revenue = float(result["total_revenue_including_afrr_capacity_eur"][0])
    elif "total_revenue_including_afrr_eur" in result:
        total_revenue = float(result["total_revenue_including_afrr_eur"][0])
    else:
        total_revenue = float(result["total_revenue"][0])

    pure_pv_revenue = float(pure_pv_benchmark["total_pv_only_revenue_eur"][0])
    hybrid_added_value = total_revenue - pure_pv_revenue

    pv_rev_keur_per_mw = pv_revenue / max(pv_dc_mw, 1e-12) / 1000.0
    bess_rev_keur_per_mw = bess_revenue_total / max(batt_power_mw, 1e-12) / 1000.0

    pv_sold_mwh = float(result["pv_direct_sold_mwh"][0])
    bess_sold_mwh = float(result["energy_shifted_mwh"][0])

    afrr_discharged_mwh = float(np.sum(result["afrr_discharge_hourly_mwh"])) if "afrr_discharge_hourly_mwh" in result else 0.0
    bess_grid_charged_mwh = float(np.sum(result["grid_charge"])) if "grid_charge" in result else 0.0
    afrr_charged_mwh = float(np.sum(result["afrr_charge_hourly_mwh"])) if "afrr_charge_hourly_mwh" in result else 0.0
    bess_total_discharged_mwh = bess_sold_mwh + afrr_discharged_mwh
    bess_total_charged_mwh = bess_grid_charged_mwh + afrr_charged_mwh
    bess_total_throughput_mwh = bess_total_charged_mwh + bess_total_discharged_mwh

    pv_rev_eur_per_mwh = pv_revenue / max(pv_sold_mwh, 1e-12)
    bess_rev_eur_per_mwh = bess_revenue_total / max(bess_total_discharged_mwh, 1e-12)

    tso_dso_curtailed = float(np.sum(curtailment_outputs["tso_dso_curtailed_mwh"]))
    self_curtailed = float(np.sum(curtailment_outputs["self_curtailed_mwh"]))
    candidate_curtailed = float(np.sum(curtailment_outputs["pv_curtailment_candidate_mwh"]))
    recovered_to_battery = float(np.sum(curtailment_outputs["pv_curtailed_to_battery_mwh_actual"]))
    residual_lost = float(np.sum(curtailment_outputs["pv_curtailed_residual_lost_mwh"]))
    max_cycles_per_year = float(result["max_cycles_per_year"][0]) if "max_cycles_per_year" in result else np.nan
    annual_discharge_cap_mwh = float(result["annual_discharge_cap_mwh"][0]) if "annual_discharge_cap_mwh" in result else np.nan
    remaining_cycle_budget_mwh = float(result["remaining_cycle_budget_mwh"][0]) if "remaining_cycle_budget_mwh" in result else np.nan
    avg_raw_bess_sell_price = float(result["avg_raw_bess_sell_price_eur_per_mwh"][0]) if "avg_raw_bess_sell_price_eur_per_mwh" in result else np.nan
    avg_effective_bess_sell_price = float(result["avg_effective_bess_sell_price_eur_per_mwh"][0]) if "avg_effective_bess_sell_price_eur_per_mwh" in result else np.nan
    revenue_loss_capture_rate = float(result["bess_revenue_loss_due_to_capture_rate_eur"][0]) if "bess_revenue_loss_due_to_capture_rate_eur" in result else np.nan
    gross_bess_revenue_before_cycle_cost = float(result["gross_bess_revenue_before_cycle_cost_eur"][0]) if "gross_bess_revenue_before_cycle_cost_eur" in result else float(result["total_batt_sale_revenue"][0])
    net_bess_revenue_after_cycle_cost = float(result["net_bess_revenue_after_cycle_cost_eur"][0]) if "net_bess_revenue_after_cycle_cost_eur" in result else bess_revenue_base
    avg_cycle_cost_per_discharged_mwh = total_cycle_cost / max(float(result["energy_shifted_mwh"][0]), 1e-12)
    cycles_without_cycle_cost = float(result["equivalent_cycles_without_cycle_cost"][0]) if "equivalent_cycles_without_cycle_cost" in result else np.nan
    cycles_with_cycle_cost = float(result["equivalent_cycles"][0]) if "equivalent_cycles" in result else np.nan

    rows = [
        ("PV Capture Rate", pv_capture_rate_pct, "%"),
        ("BESS Capture Rate", bess_capture_rate_pct, "%"),
        ("Average Raw BESS Sell Price", avg_raw_bess_sell_price, "€/MWh"),
        ("Average Effective BESS Sell Price", avg_effective_bess_sell_price, "€/MWh"),
        ("Revenue loss due to BESS capture rate", revenue_loss_capture_rate, "€"),
        ("Theoretical Cycle Cost (not deducted from cash revenue)", total_cycle_cost, "€"),
        ("Theoretical Cycle Cost per Year (not deducted)", total_cycle_cost, "€/an"),
        ("Gross BESS Revenue Before Cycle Cost", gross_bess_revenue_before_cycle_cost, "€"),
        ("BESS Cash Revenue (cycle cost not deducted)", net_bess_revenue_after_cycle_cost, "€"),
        ("Average Theoretical Cycle Cost per Discharged MWh", avg_cycle_cost_per_discharged_mwh, "€/MWh"),
        ("Number of cycles without cycle cost", cycles_without_cycle_cost, "cycles/an"),
        ("Number of cycles with cycle cost", cycles_with_cycle_cost, "cycles/an"),
        ("Revenu total", total_revenue, "€"),
        ("Revenu PV-only Project", pure_pv_revenue, "€"),
        ("Valeur ajoutée de l'hybridation vs PV-only", hybrid_added_value, "€"),
        ("Revenu PV direct", pv_revenue, "€"),
        ("Revenu batterie wholesale", float(result["total_batt_sale_revenue"][0]), "€"),
        ("Coût charge réseau wholesale", float(result["total_grid_charge_cost"][0]), "€"),
        ("Coût cycle wholesale théorique (non déduit)", wholesale_cycle_cost, "€"),
        ("Revenu services système de nuit", float(result["nightly_revenue_total"][0]), "€"),
        ("Revenu brut aFRR", afrr_sale_revenue, "€"),
        ("Cashflow charge aFRR", afrr_charge_cost, "€"),
        ("Coût cycle aFRR théorique (non déduit)", afrr_cycle_cost, "€"),
        ("Revenu net aFRR", afrr_net_revenue, "€"),
        ("Revenu aFRR Capacity UP", afrr_capacity_up_revenue, "€"),
        ("Revenu aFRR Capacity Down", afrr_capacity_down_revenue, "€"),
        ("Revenu total aFRR Capacity", afrr_capacity_revenue, "€"),
        ("Certified Capacity UP MW", certified_up_mw, "MW"),
        ("Certified Capacity Down MW", certified_down_mw, "MW"),
        ("Number of hours awarded UP", capacity_up_hours, "h"),
        ("Number of hours awarded Down", capacity_down_hours, "h"),
        ("TSO/DSO curtailed energy", tso_dso_curtailed, "MWh"),
        ("Self-curtailed energy", self_curtailed, "MWh"),
        ("Total curtailed PV candidate energy", candidate_curtailed, "MWh"),
        ("Curtailed PV recovered by battery", recovered_to_battery, "MWh"),
        ("Residual curtailed PV energy lost", residual_lost, "MWh"),
        ("Revenu PV spécifique", pv_rev_keur_per_mw, "k€/MW"),
        ("Revenu BESS spécifique", bess_rev_keur_per_mw, "k€/MW"),
        ("Revenu PV spécifique énergie", pv_rev_eur_per_mwh, "€/MWh"),
        ("Revenu BESS spécifique énergie", bess_rev_eur_per_mwh, "€/MWh"),
        ("Énergie totale vendue", float(result["energy_sold_total_mwh"][0]) + afrr_discharged_mwh, "MWh"),
        ("Énergie shiftée wholesale", bess_sold_mwh, "MWh"),
        ("Énergie déchargée aFRR", afrr_discharged_mwh, "MWh"),
        ("Énergie chargée BESS depuis réseau", bess_grid_charged_mwh, "MWh"),
        ("Énergie chargée aFRR", afrr_charged_mwh, "MWh"),
        ("Total BESS throughput", bess_total_throughput_mwh, "MWh"),
        ("Énergie PV vendue directement", pv_sold_mwh, "MWh"),
        ("Cycles équivalents batterie", float(result["equivalent_cycles"][0]), "cycles/an"),
        ("Cycles max / an", max_cycles_per_year, "cycles/an"),
        ("Annual discharge cap MWh", annual_discharge_cap_mwh, "MWh"),
        ("Remaining cycle budget", remaining_cycle_budget_mwh, "MWh"),
        ("Production PV théorique brute", float(pv_stats["annual_dc_mwh"]), "MWh"),
        ("Production PV nette valorisable", float(pv_stats["annual_net_mwh"]), "MWh"),
        ("Énergie PV perdue (pertes + disponibilité)", float(pv_stats["annual_losses_mwh"]), "MWh"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Unité"])

def format_synthese_number(value):
    """Format Synthèse numeric values with French-style space thousands separators.

    Rules:
    - Values with absolute value >= 1 000 are rounded to the nearest whole number
      and displayed without decimal places.
    - Whole numbers below 1 000 are displayed without decimal places.
    - Non-whole decimal numbers below 1 000 are displayed with max 1 decimal place.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)

        if abs(numeric_value) >= 1000:
            return f"{int(round(numeric_value)):,}".replace(",", " ")

        if abs(numeric_value - round(numeric_value)) < 1e-9:
            return f"{int(round(numeric_value))}"

        return f"{numeric_value:.1f}"

    return value

def format_synthese_table_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy of the Synthèse table with formatted numeric values."""
    display_df = summary_df.copy()
    if "Valeur" in display_df.columns:
        display_df["Valeur"] = display_df["Valeur"].apply(format_synthese_number)
    return display_df
