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

def build_effective_dispatch_prices(inputs: SimulationInputs) -> dict:
    """Build net price arrays for dispatch optimization.

    Only marginal/variable costs are included here. Corporate tax,
    withholding tax and fixed/local taxes are reporting-only and must not
    affect dispatch decisions.
    """
    pv_price_net = (
        _validate_array_length(inputs.pv_price, "PV price")
        - inputs.grid_export_fee_eur_per_mwh
        - inputs.omie_sell_fee_eur_per_mwh
        - inputs.ree_system_fee_eur_per_mwh
        - inputs.imbalance_cost_pv_eur_per_mwh
    )
    if inputs.apply_ivpee_to_pv:
        pv_price_net = pv_price_net * (1.0 - inputs.ivpee_generation_tax_pct / 100.0)

    bess_sell_price_net = (
        _validate_array_length(inputs.batt_sell_price, "BESS sell price")
        - inputs.grid_export_fee_eur_per_mwh
        - inputs.omie_sell_fee_eur_per_mwh
        - inputs.ree_system_fee_eur_per_mwh
        - inputs.imbalance_cost_bess_eur_per_mwh
    )
    if inputs.apply_ivpee_to_bess_export:
        bess_sell_price_net = bess_sell_price_net * (1.0 - inputs.ivpee_generation_tax_pct / 100.0)

    grid_buy_price_net = (
        _validate_array_length(inputs.grid_buy_price, "Grid buy price")
        + inputs.grid_import_fee_eur_per_mwh
        + inputs.omie_buy_fee_eur_per_mwh
        + inputs.ree_system_fee_eur_per_mwh
        + inputs.imbalance_cost_bess_eur_per_mwh
    )

    afrr_up_energy_price_net = None
    if inputs.afrr_discharge_price_qh is not None:
        afrr_up_energy_price_net = (
            _validate_array_length(inputs.afrr_discharge_price_qh, "aFRR UP energy price")
            * (1.0 - inputs.afrr_energy_fee_pct / 100.0)
            - inputs.afrr_energy_fee_eur_per_mwh
            - inputs.imbalance_cost_bess_eur_per_mwh
        )
        if inputs.apply_ivpee_to_afrr_energy:
            afrr_up_energy_price_net = afrr_up_energy_price_net * (
                1.0 - inputs.ivpee_generation_tax_pct / 100.0
            )

    afrr_down_energy_price_net = None
    if inputs.afrr_charge_price_qh is not None:
        # Positive DOWN price = cost to charge. Negative DOWN price = charging benefit.
        # Variable fees make DOWN charging less attractive.
        afrr_down_energy_price_net = (
            _validate_array_length(inputs.afrr_charge_price_qh, "aFRR DOWN energy price")
            + inputs.afrr_energy_fee_eur_per_mwh
            + inputs.grid_import_fee_eur_per_mwh
            + inputs.imbalance_cost_bess_eur_per_mwh
        )

    afrr_capacity_up_price_net = None
    if inputs.afrr_capacity_up_price_h is not None:
        afrr_capacity_up_price_net = (
            _validate_array_length(inputs.afrr_capacity_up_price_h, "aFRR UP capacity price")
            * (1.0 - inputs.afrr_capacity_fee_pct / 100.0)
        )
        if inputs.apply_ivpee_to_afrr_capacity:
            afrr_capacity_up_price_net = afrr_capacity_up_price_net * (
                1.0 - inputs.ivpee_generation_tax_pct / 100.0
            )

    afrr_capacity_down_price_net = None
    if inputs.afrr_capacity_down_price_h is not None:
        afrr_capacity_down_price_net = (
            _validate_array_length(inputs.afrr_capacity_down_price_h, "aFRR DOWN capacity price")
            * (1.0 - inputs.afrr_capacity_fee_pct / 100.0)
        )
        if inputs.apply_ivpee_to_afrr_capacity:
            afrr_capacity_down_price_net = afrr_capacity_down_price_net * (
                1.0 - inputs.ivpee_generation_tax_pct / 100.0
            )

    return {
        "pv_price_net": pv_price_net,
        "bess_sell_price_net": bess_sell_price_net,
        "grid_buy_price_net": grid_buy_price_net,
        "afrr_up_energy_price_net": afrr_up_energy_price_net,
        "afrr_down_energy_price_net": afrr_down_energy_price_net,
        "afrr_capacity_up_price_net": afrr_capacity_up_price_net,
        "afrr_capacity_down_price_net": afrr_capacity_down_price_net,
    }
