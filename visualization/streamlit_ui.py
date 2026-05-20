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
from data.loaders import *
from data.solar_profile import *
from data.curtailment import *
from finance.pricing import *
from finance.taxes import *
from finance.benchmarks import *
from finance.revenues import *
from optimization.dispatch_optimizer import *
from optimization.afrr import *
from optimization.afrr_capacity import *
from optimization.afrr_afry_heuristic import simulate_afrr_afry_heuristic
from optimization.afrr_milp_optimizer import simulate_afrr_milp_optimization
from optimization.reconciliation import *
from optimization.forward_curves import *
from exports.excel_export import *
from utils.dataframe_utils import *
from utils.math_utils import *
from utils.time_utils import *
from config import _open_builtin_file
from data.loaders import _read_single_column_csv, _read_single_column_csv_qh
from utils.dataframe_utils import _make_qh_dataframe
from utils.math_utils import _make_flat_curve

def run_app():
    st.set_page_config(page_title="Évaluation revenus projet hybride PV + BESS", layout="wide")
    st.title("Évaluation des revenus d'un projet hybride PV + batterie")
    st.caption("Simulation 15 minutes (35040 pas) avec optimisation économique annuelle de la batterie + co-optimisation aFRR quart-horaire Phase 1.")

    with st.expander("Hypothèses structurantes", expanded=False):
        st.markdown(
            """
            - Simulation **quart-horaire sur 35040 pas** pour le cœur du dispatch PV + BESS.
            - La batterie peut **charger depuis le PV et/ou depuis le réseau**.
            - Le moteur choisit la meilleure valorisation économique entre vente immédiate du PV, stockage PV et charge réseau.
            - Les **revenus de services système la nuit** sont ajoutés comme un **revenu fixe par nuit**, sans contrainte de capacité ni de SOC.
            - L'optimisation principale utilise une **programmation dynamique discrétisée sur le SOC**.
            - Une couche **aFRR quart-horaire Phase 1** compare wholesale, aFRR UP, aFRR DOWN et no action à chaque pas de 15 minutes.
            - La curtailment PV peut être:
              1. imposée par TSO/DSO
              2. auto-courtailment selon structure commerciale
            - Option supplémentaire: **Charge Battery if Curtailment**
              pour récupérer une partie de l'énergie autrement curtailed dans la batterie.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("BESS Parameters")
        batt_power_mw = st.number_input("BESS Usable Power (MW)", min_value=0.0, value=50.0, step=1.0)
        bess_duration_h = st.number_input("BESS Duration (h)", min_value=0.0, value=4.0, step=0.25)
        batt_energy_mwh = batt_power_mw * bess_duration_h
        st.number_input("BESS Usable Capacity (MWh)", min_value=0.0, value=float(batt_energy_mwh), step=1.0, disabled=True)
        technical_eta_charge = st.number_input("BESS Charging Efficiency (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        technical_eta_discharge = st.number_input("BESS Discharging Efficiency (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        bess_gross_capacity_mwh = batt_energy_mwh / max(technical_eta_charge * technical_eta_discharge, 1e-12)
        st.number_input("BESS Gross Capacity (MWh)", min_value=0.0, value=float(bess_gross_capacity_mwh), step=1.0, disabled=True)
        # The dispatch model uses BESS Usable Capacity directly. Technical efficiencies are used only
        # to size/report gross capacity, avoiding round-trip efficiency double-counting in revenues/spreads.
        eta_charge = 1.0
        eta_discharge = 1.0
        bess_availability_pct = st.number_input("BESS Availability (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        effective_batt_energy_mwh = batt_energy_mwh * bess_availability_pct / 100.0
        st.caption(f"Effective BESS usable capacity: {effective_batt_energy_mwh:.2f} MWh")
        min_soc_pct = st.slider("BESS Minimum SOC (%)", 0, 100, 0)
        max_soc_pct = st.slider("BESS Maximum SOC (%)", 0, 100, 100)
        initial_soc = st.number_input("BESS BoL SOH (MWh)", min_value=0.0, value=effective_batt_energy_mwh*95/100, step=1.0)
        final_soc = st.number_input("BESS EoL SOH (MWh)", min_value=0.0, value=effective_batt_energy_mwh*60/100, step=1.0)
        bess_capture_rate_pct = st.number_input("BESS Wholesale Capture Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        max_cycles_per_year = st.number_input("Max Cycles / year", min_value=0.0, value=547.0, step=0.1)
        cycle_cost = st.number_input("BESS Cycle Cost (EUR/MWh)", value=0.0)
        charge_quantile = st.slider("Charge Percentile (%)", 0, 100, 100)
        discharge_quantile = st.slider("Discharge Percentile (%)", 0, 100, 0)
        min_spread_arbitrage = st.number_input("Minimum Spread for Arbitrage (EUR/MWh)", min_value=0.0, value=80.0, step=1.0)
        nightly_bess_revenue = st.number_input("Ancillary Services Revenues (EUR/nuit)", min_value=0.0, value=0.0, step=10.0)
        
    with col2:
        st.subheader("PV Parameters")
        pv_dc_mw = st.number_input("PV DC Power (MWc)", min_value=0.0, value=100.0, step=1.0)
        productible = st.number_input("PV Yield (kWh/kWc/an)", min_value=0.0, value=1500.0, step=10.0)
        availability_pct = st.number_input("PV Availability (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        pv_losses_pct = st.number_input("PV System Losses (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        pv_capture_rate_pct = st.number_input("PV Capture Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        
    with col3:
        st.subheader("General Parameters")
        project_lifetime_years = int(st.number_input("Project Lifetime (years)", min_value=1, value=1, step=1))
        grid_export_limit_mw = st.number_input("Grid Injection Limit (MW)", min_value=0.0, value=100.0, step=1.0)
        soc_steps = st.slider("SOC Steps for Optimization", min_value=21, max_value=201, value=21, step=10)

    bess_degradation_upload = st.file_uploader(
            "BESS Degradation Curve",
            type=["xlsx", "xls", "csv"],
            key="bess_degradation_curve",
        )

    st.subheader("PV Commercial Structure")

    contract_col1, contract_col2, contract_col3 = st.columns(3)

    with contract_col1:
        enable_cfd = st.radio("CfD", ["No", "Yes"], horizontal=True) == "Yes"
        cfd_price_standalone = 0.0
        if enable_cfd:
            cfd_price_standalone = st.number_input("CfD Price (€/MWh)", value=50.0, step=1.0)

    with contract_col2:
        enable_ppa = st.radio("PPA", ["No", "Yes"], horizontal=True) == "Yes"
        ppa_price_standalone = 0.0
        if enable_ppa:
            ppa_price_standalone = st.number_input("PPA Price (€/MWh)", value=50.0, step=1.0)
            
    st.subheader("Courbe solaire 15 min (35040 pas)")
    solar_mode = st.radio("Source du profil solaire", ["Courbe standard France", "Upload CSV 35040"], horizontal=True)

    solar_upload = None
    uploaded_solar_is_relative = True
    if solar_mode == "Upload CSV 35040":
        solar_upload = st.file_uploader("Upload du profil solaire CSV (35040 lignes, première colonne numérique)", type=["xlsx", "xls", "csv"], key="solar_csv")
        uploaded_solar_is_relative = st.checkbox(
            "Le CSV uploadé est un profil relatif à normaliser sur le productible annuel (sinon : MWh nets 15 minutes absolus)",
            value=True,
        )

    st.subheader("Sell Price - PV/Grid")
    pv_price_mode = st.radio(
        "Source du prix de vente du PV",
        ["Prix moyen annuel", "Upload CSV 35040"],
        index=1,
        horizontal=True,
    )
    pv_price_value = None
    pv_price_upload = None
    if pv_price_mode == "Prix moyen annuel":
        pv_price_value = st.number_input("Prix moyen PV (EUR/MWh)", value=55.0, step=1.0)
    elif pv_price_mode == "Upload CSV 35040":
        pv_price_upload = st.file_uploader("Upload prix PV CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="pv_price")

    st.subheader("Sell Price - BESS/Grid")
    batt_sell_mode = st.radio(
        "Source du prix de vente de l'énergie shiftée",
        ["Prix moyen annuel", "Upload CSV 35040"],
        index=1,
        horizontal=True,
    )
    batt_sell_value = None
    batt_sell_upload = None
    if batt_sell_mode == "Prix moyen annuel":
        batt_sell_value = st.number_input("Prix moyen vente batterie (EUR/MWh)", value=90.0, step=1.0)
    elif batt_sell_mode == "Upload CSV 35040":
        batt_sell_upload = st.file_uploader("Upload prix vente batterie CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="batt_sell")

    st.subheader("Buy Price - BESS/Grid")
    grid_mode = st.radio(
        "Source du prix d'achat réseau",
        ["Identique au prix vente batterie", "Prix moyen annuel", "Upload CSV 35040"],
        index=2,
        horizontal=True,
    )
    grid_buy_value = None
    grid_buy_upload = None
    if grid_mode == "Prix moyen annuel":
        grid_buy_value = st.number_input("Prix moyen achat réseau (EUR/MWh)", value=55.0, step=1.0)
    elif grid_mode == "Upload CSV 35040":
        grid_buy_upload = st.file_uploader("Upload prix achat réseau CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="grid_buy")

    st.subheader("Curtailment")
    cur1, cur2, cur3 = st.columns(3)

    with cur1:
        tso_dso_curtailment = st.radio("TSO/DSO Curtailment", ["No", "Yes"], index=1, horizontal=True)
        tso_dso_upload = None
        tso_dso_source = "Curtailment Curve"
        if tso_dso_curtailment == "Yes":
            tso_dso_source = st.radio(
                "Source de la courbe TSO/DSO",
                ["Curtailment Curve", "Upload Annual Curtailment Curve Excel (12 monthly %)"],
                horizontal=False,
            )
            if tso_dso_source == "Upload Annual Curtailment Curve Excel (12 monthly %)":
                tso_dso_upload = st.file_uploader("Upload Annual Curtailment Curve Excel (12 monthly %)", type=["xlsx", "xls", "csv"], key="tso_dso_curve")

    with cur2:
        self_curtailment = st.radio("Self Curtailment", ["No", "Yes"], horizontal=True)
        curtailment_threshold = -1.0
        pv_structure = "Fully merchant"
        cfd_price = 0.0
        negative_price_rule = False
        consecutive_negative_hours_limit = 6
        ppa_price = 0.0

        if self_curtailment == "Yes":
            curtailment_threshold = st.number_input("Curtailment Threshold (EUR/MWh)", value=-1.0, step=1.0)
            pv_structure = st.radio("PV Commercial Structure", ["Fully merchant", "With CfD", "With PPA"], horizontal=False)

            if pv_structure == "With CfD":
                cfd_price = st.number_input("CfD Price (EUR/MWh)", value=50.0, step=1.0)
                negative_price_rule_str = st.radio("Negative Price Rule", ["No", "Yes"], horizontal=True)
                negative_price_rule = negative_price_rule_str == "Yes"
                if negative_price_rule:
                    consecutive_negative_hours_limit = int(st.number_input("Consecutive Negative Hours Limit", min_value=1, value=6, step=1))

            if pv_structure == "With PPA":
                ppa_price = st.number_input("PPA Price (EUR/MWh)", value=50.0, step=1.0)

    with cur3:
        charge_battery_if_curtailment = st.radio("Charge Battery if Curtailment", ["Yes", "No"], horizontal=True) == "Yes"
        
    st.subheader("aFRR Optimization")
    afrr_optimization_method = st.selectbox(
        "aFRR Optimization Method",
        ["Current model", "AFRY-style heuristic", "Full MILP optimization"],
        index=0,
        help=(
            "Current model: existing internal dispatch. "
            "AFRY-style heuristic: fast expected-value block screening. "
            "Full MILP optimization: rigorous method selector with block-level audit; may be slower."
        ),
    )
    if afrr_optimization_method == "Full MILP optimization":
        st.warning("Full MILP optimization can take longer on a full 35,040-step year. This version keeps the existing SOC deliverability engine and adds MILP audit fields.")

    st.subheader("aFRR Capacity")
    enable_afrr_capacity = st.checkbox("Activer aFRR Capacity", value=False)

    afrr_capacity_up_upload = None
    afrr_capacity_down_upload = None
    afrr_certified_capacity_pct = 100.0
    afrr_capacity_success_rate_pct = 80.0
    afrr_capacity_start_hour = 20
    afrr_capacity_end_hour = 8

    if enable_afrr_capacity:
        cap_col1, cap_col2, cap_col3 = st.columns(3)

        with cap_col1:
            st.caption("aFRR Capacity datasets are no longer embedded. Upload 35040-step files.")
            afrr_capacity_up_upload = st.file_uploader(
                "Upload afrr_up_capacity_price_15min_spain_2025 Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_capacity_up",
            )
            afrr_capacity_down_upload = st.file_uploader(
                "Upload afrr_down_capacity_price_15min_spain_2025 Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_capacity_down",
            )

        with cap_col2:
            afrr_certified_capacity_pct = st.number_input(
                "% of Certified Capacity for aFRR",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
            )
            afrr_capacity_success_rate_pct = st.slider(
                "aFRR Capacity Bid Success Rate (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=1,
            )
            st.caption("Used only in expected-value optimization; it does not reduce physical MW/MWh dispatch.")

        with cap_col3:
            afrr_capacity_start_hour = st.slider(
                "aFRR Capacity start hour",
                min_value=0,
                max_value=23,
                value=20,
                step=1,
            )
            afrr_capacity_end_hour = st.slider(
                "aFRR Capacity end hour",
                min_value=0,
                max_value=23,
                value=8,
                step=1,
            )
            st.caption("Allowed Capacity window. Overnight windows are supported, e.g. 20 → 8.")

    st.subheader("aFRR Energy")
    enable_afrr = st.checkbox("Activer aFRR Energy", value=False)
    allow_afrr_energy_without_capacity = st.checkbox(
        "Allow aFRR energy without aFRR capacity",
        value=True,
    )

    afrr_charge_upload = None
    afrr_discharge_upload = None
    afrr_min_spread = 0.0
    afrr_cycle_cost = cycle_cost
    afrr_night_start_hour = 20
    afrr_night_end_hour = 8
    afrr_max_events_per_day = 1
    afrr_energy_down_activation_pct = 100.0
    afrr_energy_up_activation_pct = 100.0
    forward_optimization_horizon_hours = 24.0
    afrr_up_cross_market_min_spread = 20.0
    afrr_down_to_wholesale_min_spread = 20.0

    if enable_afrr:
        c_afrr1, c_afrr2, c_afrr3 = st.columns(3)

        with c_afrr1:
            st.caption("aFRR Energy datasets are no longer embedded. Upload 35040-step files.")
            afrr_charge_upload = st.file_uploader(
                "Upload prix aFRR charge / down energy Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_charge",
            )
            afrr_discharge_upload = st.file_uploader(
                "Upload prix aFRR décharge / up energy Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_discharge",
            )

        with c_afrr2:
            afrr_min_spread = st.number_input("Spread minimum aFRR net (EUR/MWh)", min_value=0.0, value=min_spread_arbitrage, step=1.0)
            afrr_cycle_cost = st.number_input("Coût de cycle aFRR (EUR/MWh)", min_value=0.0, value=float(cycle_cost), step=1.0)
            afrr_energy_down_activation_pct = st.number_input("aFRR Energy Down Activation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            afrr_energy_up_activation_pct = st.number_input("aFRR Energy Up Activation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

        with c_afrr3:
            afrr_night_start_hour = st.slider(
                "aFRR Energy start hour",
                min_value=0,
                max_value=23,
                value=20,
                step=1,
            )
            afrr_night_end_hour = st.slider(
                "aFRR Energy end hour",
                min_value=0,
                max_value=23,
                value=8,
                step=1,
            )
            st.caption("Allowed Energy window. Overnight windows are supported, e.g. 20 → 8.")
            forward_optimization_horizon_hours = st.slider("Forward Optimization Horizon (hours)", min_value=1, max_value=72, value=72, step=1)
            afrr_up_cross_market_min_spread = st.number_input("Minimum Spread Wholesale Charge → aFRR UP Discharge (€/MWh)", min_value=0.0, value=min_spread_arbitrage, step=1.0)
            afrr_down_to_wholesale_min_spread = st.number_input("Minimum Spread aFRR DOWN Charge → Wholesale Discharge (€/MWh)", min_value=0.0, value=min_spread_arbitrage, step=1.0)
            afrr_max_events_per_day = st.number_input("Nombre max d'événements aFRR / jour (legacy, not used in Phase 1 capacity mode)", min_value=1, value=1, step=1)

    with st.expander("Spain taxes, grid fees and market fees", expanded=False):
        st.caption(
            "Marginal/variable fees are included in dispatch-effective prices. "
            "Corporate tax, withholding tax and local/fixed taxes are reporting-only."
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            grid_import_fee_eur_per_mwh = st.number_input("Grid import fee for BESS charging (€/MWh)", min_value=0.0, value=0.0, step=0.1)
            grid_export_fee_eur_per_mwh = st.number_input("Grid export fee (€/MWh)", min_value=0.0, value=0.0, step=0.1)
            omie_buy_fee_eur_per_mwh = st.number_input("OMIE / market fee on DA purchases (€/MWh)", min_value=0.0, value=0.0, step=0.01)
            omie_sell_fee_eur_per_mwh = st.number_input("OMIE / market fee on DA sales (€/MWh)", min_value=0.0, value=0.0, step=0.01)
            ree_system_fee_eur_per_mwh = st.number_input("REE / system operator variable fee (€/MWh)", min_value=0.0, value=0.0, step=0.01)
        with f2:
            imbalance_cost_pv_eur_per_mwh = st.number_input("PV imbalance / deviation cost (€/MWh)", min_value=0.0, value=0.0, step=0.1)
            imbalance_cost_bess_eur_per_mwh = st.number_input("BESS imbalance / deviation cost (€/MWh)", min_value=0.0, value=0.0, step=0.1)
            afrr_capacity_fee_pct = st.number_input("aFRR capacity fee (% of capacity revenue)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            afrr_energy_fee_pct = st.number_input("aFRR energy fee (% of energy revenue)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            afrr_energy_fee_eur_per_mwh = st.number_input("aFRR energy variable fee (€/MWh)", min_value=0.0, value=0.0, step=0.1)
        with f3:
            ivpee_generation_tax_pct = st.number_input("IVPEE / electricity generation tax (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            apply_ivpee_to_pv = st.checkbox("Apply IVPEE to PV revenue", value=True)
            apply_ivpee_to_bess_export = st.checkbox("Apply IVPEE to BESS DA export revenue", value=False)
            apply_ivpee_to_afrr_energy = st.checkbox("Apply IVPEE to aFRR energy revenue", value=False)
            apply_ivpee_to_afrr_capacity = st.checkbox("Apply IVPEE to aFRR capacity revenue", value=False)
            corporate_tax_pct = st.number_input("Corporate income tax (%) - financial model only", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            withholding_tax_pct = st.number_input("Withholding tax (%) - financial model only", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            local_fixed_tax_eur_per_year = st.number_input("Local / fixed taxes (€/year)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    generate_excel_export = st.checkbox(
        "Créer le fichier Excel complet à la fin de la simulation",
        value=False,
        help=(
            "Décochez cette option pour accélérer la simulation. "
            "Le fichier Excel complet contient beaucoup de lignes et peut prendre du temps à générer. "
            "Les résultats affichés dans Streamlit et les exports CSV restent disponibles."
        ),
    )
    run = st.button("Lancer la simulation", type="primary")

    if not run:
        return

    start_time = time.time()

    try:
        if effective_batt_energy_mwh < batt_power_mw and effective_batt_energy_mwh > 0:
            st.warning("Attention : la capacité batterie est inférieure à 1h de puissance. C'est possible, mais atypique.")
        if initial_soc > effective_batt_energy_mwh:
            st.error("Le SOC initial ne peut pas dépasser la capacité batterie.")
            return
        if final_soc > effective_batt_energy_mwh:
            st.error("Le SOC final ne peut pas dépasser la capacité batterie.")
            return
        if not (0.0 <= min_soc_pct <= 100.0):
            st.error("Minimum SOC batterie (%) doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= max_soc_pct <= 100.0):
            st.error("Maximum SOC batterie (%) doit être compris entre 0 et 100 %.")
            return
        if min_soc_pct >= max_soc_pct:
            st.error("Minimum SOC batterie (%) doit être strictement inférieur au Maximum SOC batterie (%).")
            return

        min_soc_mwh = effective_batt_energy_mwh * min_soc_pct / 100.0
        max_soc_mwh = effective_batt_energy_mwh * max_soc_pct / 100.0

        if initial_soc < min_soc_mwh or initial_soc > max_soc_mwh:
            st.error(
                f"Le SOC initial doit être compris entre {min_soc_mwh:.2f} MWh "
                f"et {max_soc_mwh:.2f} MWh."
            )
            return
        if final_soc < min_soc_mwh or final_soc > max_soc_mwh:
            st.error(
                f"Le SOC final doit être compris entre {min_soc_mwh:.2f} MWh "
                f"et {max_soc_mwh:.2f} MWh."
            )
            return
        if enable_cfd and enable_ppa:
            st.error("CfD et PPA ne peuvent pas être activés en même temps.")
            return
        if enable_afrr_capacity and not enable_afrr:
            st.error("Veuillez activer aFRR Energy pour utiliser aFRR Capacity.")
            return
        if enable_afrr and (not enable_afrr_capacity) and (not allow_afrr_energy_without_capacity):
            st.error("La participation en aFRR Energy sans aFRR Capacity n’est pas autorisée.")
            return
        if enable_afrr_capacity:
            if afrr_capacity_up_upload is None:
                st.error("Merci d'uploader le fichier aFRR Capacity UP 15 minutes (35040 lignes).")
                return
            if afrr_capacity_down_upload is None:
                st.error("Merci d'uploader le fichier aFRR Capacity Down 15 minutes (35040 lignes).")
                return
        if not (0.0 <= afrr_certified_capacity_pct <= 100.0):
            st.error("% of Certified Capacity for aFRR doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= afrr_energy_down_activation_pct <= 100.0):
            st.error("aFRR Energy Down Activation (%) doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= afrr_energy_up_activation_pct <= 100.0):
            st.error("aFRR Energy Up Activation (%) doit être compris entre 0 et 100 %.")
            return

        try:
            bess_degradation_curve_pct, degraded_bess_energy_by_year_mwh, bess_degradation_df = read_bess_degradation_excel(
                bess_degradation_upload,
                project_lifetime_years,
                effective_batt_energy_mwh,
            )
        except Exception as e:
            st.error(f"Erreur courbe de dégradation BESS: {e}")
            return
            
        # Base PV
        if solar_mode == "Courbe standard France":
            solar_relative = build_standard_france_solar_profile()
            base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                solar_relative, pv_dc_mw, productible, pv_losses_pct, availability_pct
            )
        else:
            if solar_upload is None:
                st.error("Merci d'uploader un fichier solaire 35040 pas de 15 minutes.")
                return

            uploaded = _read_single_column_csv(solar_upload)
            if uploaded_solar_is_relative:
                base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                    uploaded, pv_dc_mw, productible, pv_losses_pct, availability_pct
                )
            else:
                base_pv_hourly_mwh = np.maximum(uploaded, 0.0) * pv_dc_mw
                annual_net = float(base_pv_hourly_mwh.sum())
                annual_dc = float(pv_dc_mw * productible)
                pv_stats = {
                    "annual_dc_mwh": annual_dc,
                    "annual_net_mwh": annual_net,
                    "annual_losses_mwh": float(max(annual_dc - annual_net, 0.0)),
                }

        # Market/aFRR datasets are not embedded anymore. Upload files when an upload source is selected.
        if (not enable_cfd) and (not enable_ppa) and pv_price_mode == "Upload CSV 35040" and pv_price_upload is None:
            st.error("Merci d'uploader le fichier de prix PV / spot 15 minutes (35040 lignes).")
            return
        if batt_sell_mode == "Upload CSV 35040" and batt_sell_upload is None:
            st.error("Merci d'uploader le fichier de prix de vente batterie 15 minutes (35040 lignes).")
            return
        if grid_mode == "Upload CSV 35040" and grid_buy_upload is None:
            st.error("Merci d'uploader le fichier de prix d'achat réseau 15 minutes (35040 lignes).")
            return

        # Raw price curves
        pv_price_curve_raw = None

        if enable_cfd:
            pv_price_curve_raw = _make_flat_curve(cfd_price_standalone)
        elif enable_ppa:
            pv_price_curve_raw = _make_flat_curve(ppa_price_standalone)
        else:
            if pv_price_mode == "Prix moyen annuel":
                pv_price_curve_raw = _make_flat_curve(pv_price_value)
            else:
                pv_price_curve_raw = _read_single_column_csv(pv_price_upload)

        if pv_price_curve_raw is None:
            raise ValueError("pv_price_curve_raw was not properly initialized.")

        if batt_sell_mode == "Prix moyen annuel":
            batt_sell_curve_raw = _make_flat_curve(batt_sell_value)
        else:
            batt_sell_curve_raw = _read_single_column_csv(batt_sell_upload)

        if grid_mode == "Identique au prix vente batterie":
            grid_buy_curve_raw = batt_sell_curve_raw.copy()
        elif grid_mode == "Prix moyen annuel":
            grid_buy_curve_raw = _make_flat_curve(grid_buy_value)
        else:
            grid_buy_curve_raw = _read_single_column_csv(grid_buy_upload)

        afrr_charge_curve_qh_raw = None
        afrr_discharge_curve_qh_raw = None
        if enable_afrr:
            if afrr_charge_upload is None:
                st.error("Merci d'uploader le fichier aFRR charge / down energy 15 minutes (35040 lignes).")
                return
            if afrr_discharge_upload is None:
                st.error("Merci d'uploader le fichier aFRR décharge / up energy 15 minutes (35040 lignes).")
                return
            afrr_charge_curve_qh_raw = _read_single_column_csv_qh(afrr_charge_upload)
            afrr_discharge_curve_qh_raw = _read_single_column_csv_qh(afrr_discharge_upload)

        # aFRR Capacity hourly prices and certified capacities
        afrr_capacity_up_price_h_raw = None
        afrr_capacity_down_price_h_raw = None
        afrr_certified_capacity_up_mw = 0.0
        afrr_certified_capacity_down_mw = 0.0

        def read_afrr_capacity_file(uploaded_file, year):
            # 2025 aFRR capacity datasets are already in 15-minute resolution: one numeric column, 35040 rows.
            return _read_single_column_csv_qh(uploaded_file)
        
        if enable_afrr_capacity:
            try:
                afrr_capacity_up_price_h_raw = read_afrr_capacity_file(afrr_capacity_up_upload, DEFAULT_YEAR)
                afrr_capacity_down_price_h_raw = read_afrr_capacity_file(afrr_capacity_down_upload, DEFAULT_YEAR)
            except Exception as e:
                st.error(f"Erreur fichier aFRR Capacity: {e}")
                return

            afrr_certified_capacity_up_mw = (
                batt_power_mw
                * afrr_certified_capacity_pct / 100.0
                * availability_pct / 100.0
                * eta_discharge
            )
            afrr_certified_capacity_down_mw = (
                batt_power_mw
                * afrr_certified_capacity_pct / 100.0
                * availability_pct / 100.0
                * eta_charge
            )

        # Capture rates
        pv_capture_factor = pv_capture_rate_pct / 100.0
        bess_capture_factor = bess_capture_rate_pct / 100.0

        pv_spot_price_effective = pv_price_curve_raw * pv_capture_factor

        # BESS Capture Rate represents imperfect monetization of discharge value only.
        # It reduces BESS sell prices used by the optimizer, but it must not reduce
        # grid charging prices, PV prices, charging energy, or physical capacity.
        batt_sell_curve_effective = batt_sell_curve_raw * bess_capture_factor
        grid_buy_curve_effective = grid_buy_curve_raw.copy()

        afrr_charge_curve_qh_effective = None
        afrr_discharge_curve_qh_effective = None
        if enable_afrr:
            # IMPORTANT: BESS Wholesale Capture Rate applies only to DA/wholesale BESS sales.
            # aFRR Energy and aFRR Capacity must remain valued from their own price curves,
            # so setting BESS Wholesale Capture Rate to 0% disables wholesale arbitrage
            # without disabling aFRR cycling or aFRR revenues.
            afrr_charge_curve_qh_effective = afrr_charge_curve_qh_raw.copy()
            afrr_discharge_curve_qh_effective = afrr_discharge_curve_qh_raw.copy()

        # 1) TSO/DSO curtailment
        if tso_dso_curtailment == "Yes":
            if tso_dso_source == "Curtailment Curve":
                with _open_builtin_file(BUILTIN_CURTAILMENT_CURVE, "Curtailment Curve") as f:
                    tso_dso_monthly_pct = read_monthly_curtailment_excel(f)
            else:
                if tso_dso_upload is None:
                    st.error("Merci d'uploader la courbe annuelle de curtailment TSO/DSO.")
                    return
                tso_dso_monthly_pct = read_monthly_curtailment_excel(tso_dso_upload)
            tso_out = apply_tso_dso_curtailment(base_pv_hourly_mwh, tso_dso_monthly_pct)
        else:
            tso_out = {
                "pv_after_tso_dso_mwh": base_pv_hourly_mwh.copy(),
                "tso_dso_curtailed_mwh": np.zeros(QH_PER_YEAR, dtype=float),
                "tso_dso_curtailment_flag": np.zeros(QH_PER_YEAR, dtype=int),
                "tso_dso_monthly_pct_hourly": np.zeros(QH_PER_YEAR, dtype=float),
            }
            tso_dso_monthly_pct = np.zeros(12, dtype=float)

        # 2) Self curtailment
        self_out = apply_self_curtailment(
            pv_hourly_mwh=tso_out["pv_after_tso_dso_mwh"],
            pv_spot_price_raw=pv_price_curve_raw,
            pv_spot_price_effective=pv_spot_price_effective,
            enable_self_curtailment=(self_curtailment == "Yes"),
            pv_commercial_structure=pv_structure,
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
        )

        if enable_cfd or enable_ppa:
            self_out["pv_effective_price_eur_per_mwh"] = pv_spot_price_effective.copy()
            
        # Curtailment pipeline
        pv_after_tso_dso = tso_out["pv_after_tso_dso_mwh"]
        pv_after_self = self_out["pv_after_self_curtailment_mwh"]
        pv_curtailment_candidate_mwh = np.maximum(base_pv_hourly_mwh - pv_after_self, 0.0)

        if charge_battery_if_curtailment:
            curtailed_pv_recoverable_mwh = pv_curtailment_candidate_mwh.copy()
        else:
            curtailed_pv_recoverable_mwh = np.zeros(QH_PER_YEAR, dtype=float)

        pv_sellable_for_dispatch_mwh = pv_after_self.copy()
        pv_effective_price_for_revenue = self_out["pv_effective_price_eur_per_mwh"]

        # PV-only benchmark uses only sellable curtailed PV
        pure_pv_benchmark = build_pure_pv_benchmark(
            pv_generation_mwh=pv_sellable_for_dispatch_mwh,
            pv_price=pv_effective_price_for_revenue,
            grid_export_limit_mw=grid_export_limit_mw,
        )
        sim_inputs = SimulationInputs(
            afrr_optimization_method=afrr_optimization_method,
            batt_power_mw=batt_power_mw,
            batt_energy_mwh=effective_batt_energy_mwh,
            nominal_batt_energy_mwh=batt_energy_mwh,
            bess_availability_pct=bess_availability_pct,
            bess_duration_h=float(bess_duration_h),
            gross_batt_energy_mwh=float(bess_gross_capacity_mwh),
            technical_eta_charge=float(technical_eta_charge),
            technical_eta_discharge=float(technical_eta_discharge),
            pv_dc_mw=pv_dc_mw,
            productible_kwh_per_kwp=productible,
            pv_losses_pct=pv_losses_pct,
            plant_availability_pct=availability_pct,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            pv_price=pv_effective_price_for_revenue,
            batt_sell_price=batt_sell_curve_effective,
            grid_buy_price=grid_buy_curve_effective,
            solar_profile=pv_sellable_for_dispatch_mwh,
            curtailed_pv_recoverable_mwh=curtailed_pv_recoverable_mwh,
            nightly_bess_revenue_eur=nightly_bess_revenue,
            soc_steps=soc_steps,
            initial_soc_mwh=initial_soc,
            final_soc_mwh=final_soc,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            grid_export_limit_mw=grid_export_limit_mw,
            cycle_cost_eur_per_mwh=cycle_cost,
            charge_quantile=charge_quantile,
            discharge_quantile=discharge_quantile,
            max_cycles_per_year=max_cycles_per_year,
            min_spread_arbitrage_eur_per_mwh=min_spread_arbitrage,
            forward_optimization_horizon_hours=float(forward_optimization_horizon_hours),
            afrr_up_cross_market_min_spread_eur_per_mwh=float(afrr_up_cross_market_min_spread),
            afrr_down_to_wholesale_min_spread_eur_per_mwh=float(afrr_down_to_wholesale_min_spread),
            pv_capture_rate_pct=pv_capture_rate_pct,
            bess_capture_rate_pct=bess_capture_rate_pct,
            grid_import_fee_eur_per_mwh=grid_import_fee_eur_per_mwh,
            grid_export_fee_eur_per_mwh=grid_export_fee_eur_per_mwh,
            omie_buy_fee_eur_per_mwh=omie_buy_fee_eur_per_mwh,
            omie_sell_fee_eur_per_mwh=omie_sell_fee_eur_per_mwh,
            ree_system_fee_eur_per_mwh=ree_system_fee_eur_per_mwh,
            imbalance_cost_pv_eur_per_mwh=imbalance_cost_pv_eur_per_mwh,
            imbalance_cost_bess_eur_per_mwh=imbalance_cost_bess_eur_per_mwh,
            afrr_capacity_fee_pct=afrr_capacity_fee_pct,
            afrr_energy_fee_pct=afrr_energy_fee_pct,
            afrr_energy_fee_eur_per_mwh=afrr_energy_fee_eur_per_mwh,
            ivpee_generation_tax_pct=ivpee_generation_tax_pct,
            apply_ivpee_to_pv=apply_ivpee_to_pv,
            apply_ivpee_to_bess_export=apply_ivpee_to_bess_export,
            apply_ivpee_to_afrr_energy=apply_ivpee_to_afrr_energy,
            apply_ivpee_to_afrr_capacity=apply_ivpee_to_afrr_capacity,
            corporate_tax_pct=corporate_tax_pct,
            withholding_tax_pct=withholding_tax_pct,
            local_fixed_tax_eur_per_year=local_fixed_tax_eur_per_year,
            enable_afrr=enable_afrr,
            afrr_charge_price_qh=afrr_charge_curve_qh_effective,
            afrr_discharge_price_qh=afrr_discharge_curve_qh_effective,
            afrr_min_spread_eur_per_mwh=afrr_min_spread,
            afrr_cycle_cost_eur_per_mwh=afrr_cycle_cost,
            afrr_max_events_per_day=int(afrr_max_events_per_day),
            afrr_night_start_hour=int(afrr_night_start_hour),
            afrr_night_end_hour=int(afrr_night_end_hour),
            afrr_pv_zero_tolerance_mwh=PV_ZERO_TOLERANCE_MWH,
            afrr_n_qh_per_side=16,
            afrr_energy_down_activation_pct=afrr_energy_down_activation_pct,
            afrr_energy_up_activation_pct=afrr_energy_up_activation_pct,
            enable_afrr_capacity=enable_afrr_capacity,
            afrr_capacity_up_price_h=afrr_capacity_up_price_h_raw,
            afrr_capacity_down_price_h=afrr_capacity_down_price_h_raw,
            afrr_certified_capacity_pct=afrr_certified_capacity_pct,
            afrr_capacity_success_rate_pct=float(afrr_capacity_success_rate_pct),
            allow_afrr_energy_without_capacity=allow_afrr_energy_without_capacity,
            afrr_certified_capacity_up_mw=afrr_certified_capacity_up_mw,
            afrr_certified_capacity_down_mw=afrr_certified_capacity_down_mw,
            afrr_capacity_start_hour=int(afrr_capacity_start_hour),
            afrr_capacity_end_hour=int(afrr_capacity_end_hour),
            enable_tso_dso_curtailment=(tso_dso_curtailment == "Yes"),
            tso_dso_monthly_curtailment_pct=tso_dso_monthly_pct,
            enable_self_curtailment=(self_curtailment == "Yes"),
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            pv_commercial_structure=pv_structure,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
            charge_battery_if_curtailment=charge_battery_if_curtailment,
            enable_cfd=enable_cfd,
            cfd_price_standalone_eur_per_mwh=cfd_price_standalone,
            enable_ppa=enable_ppa,
            ppa_price_standalone_eur_per_mwh=ppa_price_standalone,
            project_lifetime_years=project_lifetime_years,
            bess_degradation_curve_pct=bess_degradation_curve_pct,
            degraded_bess_energy_by_year_mwh=degraded_bess_energy_by_year_mwh,
        )

        # Keep a gross input object for reporting, then create a dispatch-only
        # copy using net effective prices. This avoids double-counting fees:
        # dispatch sees net prices, reporting still shows gross revenues and
        # variable fees/taxes separately.
        gross_inputs = sim_inputs
        effective_prices = build_effective_dispatch_prices(gross_inputs)
        sim_inputs = replace(
            gross_inputs,
            pv_price=effective_prices["pv_price_net"],
            batt_sell_price=effective_prices["bess_sell_price_net"],
            grid_buy_price=effective_prices["grid_buy_price_net"],
            afrr_discharge_price_qh=effective_prices["afrr_up_energy_price_net"],
            afrr_charge_price_qh=effective_prices["afrr_down_energy_price_net"],
            afrr_capacity_up_price_h=effective_prices["afrr_capacity_up_price_net"],
            afrr_capacity_down_price_h=effective_prices["afrr_capacity_down_price_net"],
        )

        # Phase 1 co-optimization flow:
        # 1) run a baseline wholesale DP without aFRR capacity blocking,
        # 2) select aFRR capacity by expected-value comparison against that wholesale reference,
        # 3) rerun the final DP with selected aFRR capacity intervals blocked from wholesale dispatch.
        inputs_df = build_inputs_dataframe(gross_inputs)

        with st.spinner("Optimisation wholesale de référence en cours..."):
            wholesale_reference_result = optimize_dispatch_dp(sim_inputs)

        if sim_inputs.afrr_optimization_method == "AFRY-style heuristic":
            afrr_capacity_result = simulate_afrr_afry_heuristic(sim_inputs, wholesale_reference_result=wholesale_reference_result)
        elif sim_inputs.afrr_optimization_method == "Full MILP optimization":
            try:
                afrr_capacity_result = simulate_afrr_milp_optimization(sim_inputs, wholesale_reference_result=wholesale_reference_result)
            except Exception as milp_error:
                st.error(f"Full MILP optimization failed: {milp_error}")
                st.stop()
        else:
            afrr_capacity_result = simulate_afrr_capacity(sim_inputs, wholesale_reference_result=wholesale_reference_result)
        sim_inputs.afrr_capacity_selected_market_h = afrr_capacity_result["afrr_capacity_selected_market_h"]
        sim_inputs.afrr_expected_up_activated_mwh_qh = afrr_capacity_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR, dtype=float))
        sim_inputs.afrr_expected_down_activated_mwh_qh = afrr_capacity_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR, dtype=float))

        with st.spinner("Optimisation économique annuelle finale en cours..."):
            result = optimize_dispatch_dp(sim_inputs)

        afrr_result = None
        reconciliation = None
        final_result = result

        if sim_inputs.enable_afrr:
            with st.spinner("Simulation aFRR quart-horaire et validation de livrabilité SOC finale en cours..."):
                afrr_result = simulate_afrr_night_arbitrage(sim_inputs, result)
                reconciliation = reconcile_wholesale_afrr_dispatch_qh(result_hourly=result, afrr_result=afrr_result, inputs=sim_inputs)
                reconciliation, _cycle_budget_stats = enforce_hard_annual_cycle_cap_on_reconciliation(
                    reconciliation,
                    sim_inputs,
                    afrr_capacity_result=afrr_capacity_result,
                )
                final_result = build_final_result_after_market_arbitration(base_result=result, reconciliation=reconciliation, inputs=sim_inputs)

                # Final-combined-SOC deliverability enforcement.
                # The first aFRR capacity pass uses a forward approximation; this loop
                # removes UP/DOWN awards that the final physical combined SOC trajectory
                # could not deliver, then reruns the final DP/aFRR dispatch.
                for _deliverability_pass in range(3):
                    afrr_capacity_result, _deliverability_stats = enforce_afrr_capacity_deliverability_from_final_dispatch(
                        afrr_capacity_result,
                        reconciliation,
                        tolerance_mwh=1e-6,
                    )

                    if (
                        _deliverability_stats.get("removed_up", 0) == 0
                        and _deliverability_stats.get("removed_down", 0) == 0
                    ):
                        break

                    sim_inputs.afrr_capacity_selected_market_h = afrr_capacity_result["afrr_capacity_selected_market_h"]
                    sim_inputs.afrr_expected_up_activated_mwh_qh = afrr_capacity_result.get(
                        "expected_up_activated_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )
                    sim_inputs.afrr_expected_down_activated_mwh_qh = afrr_capacity_result.get(
                        "expected_down_activated_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )

                    result = optimize_dispatch_dp(sim_inputs)
                    afrr_result = simulate_afrr_night_arbitrage(sim_inputs, result)
                    reconciliation = reconcile_wholesale_afrr_dispatch_qh(result_hourly=result, afrr_result=afrr_result, inputs=sim_inputs)
                    reconciliation, _cycle_budget_stats = enforce_hard_annual_cycle_cap_on_reconciliation(
                        reconciliation,
                        sim_inputs,
                        afrr_capacity_result=afrr_capacity_result,
                    )
                    final_result = build_final_result_after_market_arbitration(base_result=result, reconciliation=reconciliation, inputs=sim_inputs)

                # Store final actual shortfalls in the capacity audit arrays.
                if reconciliation is not None:
                    afrr_capacity_result["afrr_up_expected_vs_actual_shortfall_mwh"] = reconciliation.get(
                        "afrr_up_activation_shortfall_qh_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )
                    afrr_capacity_result["afrr_down_expected_vs_actual_shortfall_mwh"] = reconciliation.get(
                        "afrr_down_activation_shortfall_qh_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )

        final_result = add_afrr_capacity_to_final_result(final_result, afrr_capacity_result)

        # Recompute actual curtailed PV recovered AFTER final dispatch/reconciliation
        pv_curtailed_to_battery_actual = final_result.get(
            "pv_curtailed_to_battery",
            result["pv_curtailed_to_battery"],
        )
        
        pv_curtailed_residual_lost_mwh = np.maximum(
            pv_curtailment_candidate_mwh - pv_curtailed_to_battery_actual,
            0.0,
        )
        
        curtailment_outputs = {
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "tso_dso_curtailed_mwh": tso_out["tso_dso_curtailed_mwh"],
            "self_curtailed_mwh": self_out["self_curtailed_mwh"],
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh_actual": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "pv_effective_price_eur_per_mwh": pv_effective_price_for_revenue,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure_hourly": self_out["pv_commercial_structure_hourly"],
        }

        # BESS Wholesale Capture Rate reporting
        # Theoretical revenue uses the same wholesale discharge volumes but the raw, uncaptured DA sell prices.
        # Do not include aFRR in this capture-rate loss calculation: aFRR is intentionally
        # independent from BESS Wholesale Capture Rate.
        wholesale_theoretical_revenue_without_capture = final_result["discharge"] * batt_sell_curve_raw
        wholesale_actual_revenue_with_capture = final_result["batt_sale_revenue"]

        bess_theoretical_revenue_without_capture_hourly = wholesale_theoretical_revenue_without_capture
        bess_actual_revenue_with_capture_hourly = wholesale_actual_revenue_with_capture
        bess_revenue_loss_due_to_capture_hourly = (
            bess_theoretical_revenue_without_capture_hourly
            - bess_actual_revenue_with_capture_hourly
        )

        final_result["bess_theoretical_revenue_without_capture_hourly_eur"] = bess_theoretical_revenue_without_capture_hourly
        final_result["bess_actual_revenue_with_capture_hourly_eur"] = bess_actual_revenue_with_capture_hourly
        final_result["bess_revenue_loss_due_to_capture_rate_hourly_eur"] = bess_revenue_loss_due_to_capture_hourly
        final_result["bess_theoretical_revenue_without_capture_eur"] = np.array([float(np.sum(bess_theoretical_revenue_without_capture_hourly))])
        final_result["bess_actual_revenue_with_capture_eur"] = np.array([float(np.sum(bess_actual_revenue_with_capture_hourly))])
        final_result["bess_revenue_loss_due_to_capture_rate_eur"] = np.array([float(np.sum(bess_revenue_loss_due_to_capture_hourly))])
        final_result["avg_raw_bess_sell_price_eur_per_mwh"] = np.array([float(np.mean(batt_sell_curve_raw))])
        final_result["avg_effective_bess_sell_price_eur_per_mwh"] = np.array([float(np.mean(batt_sell_curve_effective))])

        # Cycle cost accounting and comparison vs. a zero-cycle-cost run.
        final_result["total_cycle_cost_eur"] = np.array([
            float(final_result.get("total_wholesale_cycle_cost_eur", np.array([0.0]))[0])
            + float(final_result.get("total_afrr_cycle_cost_eur", np.array([0.0]))[0])
        ])
        total_discharged_for_cycle_cost_mwh = float(final_result.get("energy_shifted_mwh", np.array([0.0]))[0])
        final_result["average_cycle_cost_per_discharged_mwh"] = np.array([
            float(final_result["total_cycle_cost_eur"][0]) / max(total_discharged_for_cycle_cost_mwh, 1e-12)
        ])
        final_result["equivalent_cycles_with_cycle_cost"] = final_result["equivalent_cycles"].copy()

        try:
            if cycle_cost > 1e-12 or afrr_cycle_cost > 1e-12:
                no_cycle_inputs = replace(
                    sim_inputs,
                    cycle_cost_eur_per_mwh=0.0,
                    afrr_cycle_cost_eur_per_mwh=0.0,
                )
                no_cycle_inputs.afrr_capacity_selected_market_h = sim_inputs.afrr_capacity_selected_market_h
                no_cycle_result = optimize_dispatch_dp(no_cycle_inputs)
                if no_cycle_inputs.enable_afrr:
                    no_cycle_afrr_result = simulate_afrr_night_arbitrage(no_cycle_inputs, no_cycle_result)
                    no_cycle_reconciliation = reconcile_wholesale_afrr_dispatch_qh(
                        result_hourly=no_cycle_result,
                        afrr_result=no_cycle_afrr_result,
                        inputs=no_cycle_inputs,
                    )
                    no_cycle_final = build_final_result_after_market_arbitration(
                        base_result=no_cycle_result,
                        reconciliation=no_cycle_reconciliation,
                        inputs=no_cycle_inputs,
                    )
                else:
                    no_cycle_final = no_cycle_result
                no_cycle_final = add_afrr_capacity_to_final_result(no_cycle_final, afrr_capacity_result)
                final_result["equivalent_cycles_without_cycle_cost"] = np.array([float(no_cycle_final["equivalent_cycles"][0])])
                final_result["energy_shifted_without_cycle_cost_mwh"] = np.array([float(no_cycle_final["energy_shifted_mwh"][0])])
            else:
                final_result["equivalent_cycles_without_cycle_cost"] = final_result["equivalent_cycles"].copy()
                final_result["energy_shifted_without_cycle_cost_mwh"] = final_result["energy_shifted_mwh"].copy()
        except Exception as e:
            final_result["equivalent_cycles_without_cycle_cost"] = np.array([np.nan])
            final_result["energy_shifted_without_cycle_cost_mwh"] = np.array([np.nan])
            st.warning(f"Impossible de calculer le scénario sans coût de cycle: {e}")

        if reconciliation is not None:
            combined_soc_hourly_end = reconciliation["combined_soc_hourly_end_mwh"]
        else:
            combined_soc_result = build_combined_soc_with_afrr(
                result_hourly=result,
                afrr_result=None,
                batt_energy_mwh=sim_inputs.batt_energy_mwh,
                initial_soc_mwh=sim_inputs.initial_soc_mwh,
                eta_charge=sim_inputs.eta_charge,
                eta_discharge=sim_inputs.eta_discharge,
                min_soc_pct=sim_inputs.min_soc_pct,
                max_soc_pct=sim_inputs.max_soc_pct,
            )

            combined_soc_hourly_end = combined_soc_result["combined_soc_hourly_end"]

        spain_fee_tax_breakdown = compute_spain_fee_tax_breakdown(
            gross_inputs=gross_inputs,
            dispatch_inputs=sim_inputs,
            result=final_result,
            reconciliation=reconciliation,
            afrr_capacity_result=afrr_capacity_result,
            pv_benchmark=pure_pv_benchmark,
        )
        spain_fee_tax_df = _make_qh_dataframe(spain_fee_tax_breakdown)
        spain_fee_tax_summary_df = summarize_spain_fee_tax_breakdown(spain_fee_tax_breakdown)

        # Expose Spain fee/tax totals in final_result for graphs and KPI reporting.
        final_result["gross_revenue_before_fees_eur"] = np.array([float(np.sum(spain_fee_tax_breakdown["gross_revenue_before_fees_eur"]))])
        final_result["total_variable_fees_and_taxes_eur"] = np.array([float(np.sum(spain_fee_tax_breakdown["total_variable_fees_and_taxes_eur"]))])
        final_result["net_revenue_after_variable_fees_eur"] = np.array([float(np.sum(spain_fee_tax_breakdown["net_revenue_after_variable_fees_eur"]))])
        final_result["ebitda_after_fixed_costs_eur"] = spain_fee_tax_breakdown["ebitda_after_fixed_costs_eur"]
        final_result["corporate_tax_eur"] = spain_fee_tax_breakdown["corporate_tax_eur"]
        final_result["withholding_tax_eur"] = spain_fee_tax_breakdown["withholding_tax_eur"]
        final_result["cash_flow_after_tax_and_withholding_eur"] = spain_fee_tax_breakdown["cash_flow_after_tax_and_withholding_eur"]

        summary_df = build_summary_table(
            final_result,
            pv_stats,
            pure_pv_benchmark,
            pv_dc_mw,
            batt_power_mw,
            pv_capture_rate_pct,
            bess_capture_rate_pct,
            curtailment_outputs,
        )
        summary_df = pd.concat([summary_df, spain_fee_tax_summary_df], ignore_index=True)

        monthly_df = monthly_dataframe(final_result, pure_pv_benchmark, pv_dc_mw, batt_power_mw, curtailment_outputs)
        
        if enable_cfd:
             monthly_df["pv_only_cfd_revenue"] = (
                 monthly_df["pv_only_direct_mwh"] * cfd_price_standalone
             )
        else:
            monthly_df["pv_only_cfd_revenue"] = np.nan

        idx = build_quarter_hour_index(DEFAULT_YEAR)
        
        # === FIX SOC to include curtailed PV ===
        min_soc_mwh = sim_inputs.batt_energy_mwh * sim_inputs.min_soc_pct / 100.0
        max_soc_mwh = sim_inputs.batt_energy_mwh * sim_inputs.max_soc_pct / 100.0

        if reconciliation is not None:
            combined_soc_hourly_end = reconciliation["combined_soc_hourly_end_mwh"]
        else:
            hourly_charge_to_soc = (
                final_result["pv_to_batt"]
                + pv_curtailed_to_battery_actual
                + final_result["grid_charge"]
            ) * sim_inputs.eta_charge
        
            hourly_discharge_from_soc = (
                final_result["discharge"]
            ) / max(sim_inputs.eta_discharge, 1e-12)
        
            soc_hourly = np.zeros(QH_PER_YEAR + 1)
            soc_hourly[0] = min(max(sim_inputs.initial_soc_mwh, min_soc_mwh), max_soc_mwh)
        
            for t in range(QH_PER_YEAR):
                soc_hourly[t + 1] = min(
                    max(
                        soc_hourly[t]
                        + hourly_charge_to_soc[t]
                        - hourly_discharge_from_soc[t],
                        min_soc_mwh,
                    ),
                    max_soc_mwh,
                )
        
            combined_soc_hourly_end = soc_hourly[1:]
            
        hourly_df = _make_qh_dataframe({
            "datetime": idx,
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure": self_out["pv_commercial_structure_hourly"],
            "pv_price_raw_eur_per_mwh": pv_price_curve_raw,
            "pv_price_effective_eur_per_mwh": pv_effective_price_for_revenue,
            "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
            "pv_only_revenue_eur": pure_pv_benchmark["pv_only_revenue_eur"],
            "battery_sell_price_raw_eur_per_mwh": batt_sell_curve_raw,
            "battery_sell_price_effective_eur_per_mwh": batt_sell_curve_effective,
            "grid_buy_price_raw_eur_per_mwh": grid_buy_curve_raw,
            "grid_buy_price_effective_eur_per_mwh": grid_buy_curve_effective,
            "pv_direct_mwh": final_result["pv_direct"],
            "pv_to_battery_mwh": final_result["pv_to_batt"],
            "grid_charge_mwh": final_result["grid_charge"],
            "battery_discharge_mwh": final_result["discharge"],
            "battery_soc_mwh_end": combined_soc_hourly_end,
            "pv_direct_revenue_eur": final_result["pv_direct_revenue"],
            "battery_sale_revenue_eur": final_result["batt_sale_revenue"],
            "bess_theoretical_revenue_without_capture_eur": final_result["bess_theoretical_revenue_without_capture_hourly_eur"],
            "bess_revenue_loss_due_to_capture_rate_eur": final_result["bess_revenue_loss_due_to_capture_rate_hourly_eur"],
            "grid_charge_cost_eur": final_result["grid_charge_cost"],
            "wholesale_cycle_cost_eur": final_result["wholesale_cycle_cost_eur"] if "wholesale_cycle_cost_eur" in final_result else np.zeros(QH_PER_YEAR),
            "avg_stored_charge_price_eur_per_mwh": final_result["avg_stored_charge_price"][1:],
            "required_discharge_price_eur_per_mwh": final_result["required_discharge_price"],
            "required_discharge_price_gate_estimate_eur_per_mwh": final_result["required_discharge_price_gate_estimate"],
            "afrr_charge_mwh": final_result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_discharge_mwh": final_result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "selected_charge_market": (pd.Series(reconciliation["selected_charge_market_qh"]).to_numpy() if reconciliation is not None and "selected_charge_market_qh" in reconciliation else np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_discharge_market": (pd.Series(reconciliation["selected_discharge_market_qh"]).to_numpy() if reconciliation is not None and "selected_discharge_market_qh" in reconciliation else np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_charge_cost_eur": final_result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_sale_revenue_eur": final_result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_cycle_cost_eur": final_result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_net_revenue_eur": final_result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_price_eur_per_mw_h": afrr_capacity_up_price_h_raw if afrr_capacity_up_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_price_eur_per_mw_h": afrr_capacity_down_price_h_raw if afrr_capacity_down_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_eligible": final_result.get("afrr_capacity_eligible_h", np.zeros(QH_PER_YEAR, dtype=int)),
            "afrr_capacity_selected_market": final_result["afrr_capacity_selected_market_h"] if "afrr_capacity_selected_market_h" in final_result else np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_afry_block_best_market": final_result.get("afrr_afry_block_best_market", np.full(QH_PER_YEAR, "", dtype=object)),
            "afrr_afry_rejection_reason": final_result.get("afrr_afry_rejection_reason", np.full(QH_PER_YEAR, "", dtype=object)),
            "afrr_afry_block_up_value_eur": final_result.get("afrr_afry_block_up_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_afry_block_down_value_eur": final_result.get("afrr_afry_block_down_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_afry_block_wholesale_value_eur": final_result.get("afrr_afry_block_wholesale_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_milp_block_status": final_result.get("afrr_milp_block_status", np.full(QH_PER_YEAR, "", dtype=object)),
            "afrr_milp_rejection_reason": final_result.get("afrr_milp_rejection_reason", np.full(QH_PER_YEAR, "", dtype=object)),
            "afrr_milp_binary_up_award": final_result.get("afrr_milp_binary_up_award", np.zeros(QH_PER_YEAR)),
            "afrr_milp_binary_down_award": final_result.get("afrr_milp_binary_down_award", np.zeros(QH_PER_YEAR)),
            "afrr_capacity_up_awarded": final_result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_capacity_down_awarded": final_result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_certified_capacity_up_mw": final_result["afrr_certified_capacity_up_mw_h"] if "afrr_certified_capacity_up_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_certified_capacity_down_mw": final_result["afrr_certified_capacity_down_mw_h"] if "afrr_certified_capacity_down_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_revenue_eur": final_result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_revenue_eur": final_result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_total_revenue_eur": final_result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "battery_blocked_by_afrr_capacity": final_result["battery_blocked_by_afrr_capacity"] if "battery_blocked_by_afrr_capacity" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "wholesale_opportunity_value_eur": final_result.get("wholesale_opportunity_value_eur", np.zeros(QH_PER_YEAR)),
            "wholesale_expected_value_after_capture_rate_eur": final_result.get("wholesale_expected_value_after_capture_rate_eur", np.zeros(QH_PER_YEAR)),
            "raw_up_capacity_revenue_eur": final_result.get("raw_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_capacity_revenue_eur": final_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "raw_down_capacity_revenue_eur": final_result.get("raw_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_down_capacity_revenue_eur": final_result.get("expected_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_activated_mwh": final_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)),
            "expected_down_activated_mwh": final_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_energy_expected_value_eur": final_result.get("afrr_up_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_energy_expected_value_eur": final_result.get("afrr_down_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_up_total_expected_value_eur": final_result.get("afrr_up_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_total_expected_value_eur": final_result.get("afrr_down_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "selected_market": final_result.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_capacity_direction": final_result.get("selected_capacity_direction", np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_capacity_success_rate_pct": final_result.get("afrr_capacity_success_rate_pct", np.zeros(QH_PER_YEAR)),
            "afrr_up_activation_pct": final_result.get("afrr_up_activation_pct", np.zeros(QH_PER_YEAR)),
            "afrr_down_activation_pct": final_result.get("afrr_down_activation_pct", np.zeros(QH_PER_YEAR)),
            "available_export_headroom_mwh": final_result.get("available_export_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_soc_headroom_mwh": final_result.get("available_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_discharge_from_soc_mwh": final_result.get("available_discharge_from_soc_mwh", np.zeros(QH_PER_YEAR)),
            "required_up_soc_reserve_mwh": final_result.get("required_up_soc_reserve_mwh", np.zeros(QH_PER_YEAR)),
            "required_down_soc_headroom_mwh": final_result.get("required_down_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "expected_degradation_cost_eur": final_result.get("expected_degradation_cost_eur", np.zeros(QH_PER_YEAR)),
            "future_best_market_value_eur_per_mwh": final_result.get("future_best_market_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_best_market_type": final_result.get("future_best_market_type", np.full(QH_PER_YEAR, "none", dtype=object)),
            "cross_market_spread_eur_per_mwh": final_result.get("cross_market_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "required_min_spread_eur_per_mwh": final_result.get("required_min_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "spread_condition_respected": final_result.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
            "charge_reason": final_result.get("charge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "discharge_reason": final_result.get("discharge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "stored_energy_cost_eur_per_mwh": final_result.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "effective_discharge_value_eur_per_mwh": final_result.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_expected_afrr_up_value_eur": final_result.get("future_expected_afrr_up_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_wholesale_value_eur": final_result.get("future_expected_wholesale_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_best_discharge_market": final_result.get("future_expected_best_discharge_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "wholesale_charge_for_future_afrr_flag": final_result.get("wholesale_charge_for_future_afrr_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_wholesale_flag": final_result.get("afrr_down_charge_for_future_wholesale_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_afrr_up_flag": final_result.get("afrr_down_charge_for_future_afrr_up_flag", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_spread_ok": final_result.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_spread_ok": final_result.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "forward_horizon_hours": final_result.get("forward_horizon_hours", np.zeros(QH_PER_YEAR)),
            "future_opportunity_selected": final_result.get("future_opportunity_selected", np.zeros(QH_PER_YEAR)),
            "forward_soc_before_capacity_selection_mwh": final_result.get("forward_soc_before_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "forward_soc_after_capacity_selection_mwh": final_result.get("forward_soc_after_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_soc_feasible": final_result.get("afrr_up_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_down_soc_feasible": final_result.get("afrr_down_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_soc": final_result.get("afrr_up_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_soc": final_result.get("afrr_down_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_final_combined_soc": final_result.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_final_combined_soc": final_result.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "afrr_down_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "annual_discharge_cap_mwh": final_result.get("annual_discharge_cap_mwh", np.full(QH_PER_YEAR, sim_inputs.max_cycles_per_year * sim_inputs.batt_energy_mwh)),
            "cumulative_battery_discharge_mwh": final_result.get("cumulative_battery_discharge_mwh", np.zeros(QH_PER_YEAR)),
            "remaining_discharge_budget_mwh": final_result.get("remaining_discharge_budget_mwh", np.zeros(QH_PER_YEAR)),
            "cycle_budget_used_pct": final_result.get("cycle_budget_used_pct", np.zeros(QH_PER_YEAR)),
            "cycle_budget_available_flag": final_result.get("cycle_budget_available_flag", np.zeros(QH_PER_YEAR)),
            "discharge_rejected_due_to_cycle_budget": final_result.get("discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_rejected_due_to_cycle_budget": final_result.get("wholesale_discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_rejected_due_to_cycle_budget": final_result.get("afrr_up_discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "afrr_up_capacity_rejected_due_to_cycle_budget": final_result.get("afrr_up_capacity_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "net_dispatch_value_eur_per_mwh": final_result.get("net_dispatch_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "cycle_budget_rank": final_result.get("cycle_budget_rank", np.zeros(QH_PER_YEAR)),
            "pv_capture_rate_pct": np.full(QH_PER_YEAR, pv_capture_rate_pct),
            "bess_capture_rate_pct": np.full(QH_PER_YEAR, bess_capture_rate_pct),
        })
        hourly_df["soc_expected_from_flows"] = (
            hourly_df["battery_soc_mwh_end"].shift(1).fillna(sim_inputs.initial_soc_mwh)
            + (
                hourly_df["pv_to_battery_mwh"]
                + hourly_df["pv_curtailed_to_battery_mwh"]
                + hourly_df["grid_charge_mwh"]
                + hourly_df["afrr_charge_mwh"]
            ) * sim_inputs.eta_charge
            - (
                hourly_df["battery_discharge_mwh"]
                + hourly_df["afrr_discharge_mwh"]
            ) / sim_inputs.eta_discharge
        )

        afrr_qh_df = None
        if reconciliation is not None:
            afrr_qh_df = _make_qh_dataframe({
                "datetime": reconciliation["datetime_qh"],
                "afrr_charge_price_raw_eur_per_mwh": afrr_charge_curve_qh_raw if afrr_charge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_charge_price_effective_eur_per_mwh": sim_inputs.afrr_charge_price_qh,
                "afrr_discharge_price_raw_eur_per_mwh": afrr_discharge_curve_qh_raw if afrr_discharge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_discharge_price_effective_eur_per_mwh": sim_inputs.afrr_discharge_price_qh,
                "afrr_energy_eligible": reconciliation.get("afrr_energy_eligible_qh", np.zeros(QH_PER_YEAR, dtype=int)),
                "afrr_charge_mwh": reconciliation["afrr_charge_qh_mwh"],
                "afrr_discharge_mwh": reconciliation["afrr_discharge_qh_mwh"],
                "expected_down_activated_mwh_from_capacity_selection": sim_inputs.afrr_expected_down_activated_mwh_qh if sim_inputs.afrr_expected_down_activated_mwh_qh is not None else np.zeros(QH_PER_YEAR),
                "expected_up_activated_mwh_from_capacity_selection": sim_inputs.afrr_expected_up_activated_mwh_qh if sim_inputs.afrr_expected_up_activated_mwh_qh is not None else np.zeros(QH_PER_YEAR),
                "afrr_down_activation_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)),
                "afrr_up_activation_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)),
                "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_qh"],
                "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_qh"],
                "selected_charge_market": reconciliation["selected_charge_market_qh"],
                "selected_charge_price_eur_per_mwh": reconciliation["selected_charge_price_qh"],
                "wholesale_grid_charge_mwh": reconciliation["wholesale_grid_charge_qh_mwh"],
                "wholesale_discharge_mwh": reconciliation["wholesale_discharge_qh_mwh"],
                "selected_discharge_channel": reconciliation["selected_discharge_channel_qh"],
                "selected_discharge_market": reconciliation["selected_discharge_market_qh"],
                "selected_discharge_price_eur_per_mwh": reconciliation["selected_discharge_price_qh"],
                "stored_energy_cost_eur_per_mwh": reconciliation.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
                "effective_discharge_value_eur_per_mwh": reconciliation.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
                "spread_condition_respected": reconciliation.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
                "wholesale_discharge_spread_ok": reconciliation.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
                "afrr_up_discharge_spread_ok": reconciliation.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
                "afrr_capacity_selected_market": reconciliation["afrr_capacity_selected_market_qh"],
                "combined_soc_mwh": reconciliation["combined_soc_qh"][1:],
                "afrr_charge_cost_eur": reconciliation["afrr_charge_cost_qh_eur"],
                "afrr_sale_revenue_eur": reconciliation["afrr_sale_revenue_qh_eur"],
                "afrr_cycle_cost_eur": reconciliation["afrr_cycle_cost_qh_eur"],
                "afrr_net_revenue_eur": reconciliation["afrr_net_revenue_qh_eur"],
                "bess_capture_rate_pct": np.full(QH_PER_YEAR, bess_capture_rate_pct),
            })

        afrr_capacity_df = _make_qh_dataframe({
            "datetime": idx,
            "afrr_optimization_method": final_result.get("afrr_optimization_method", np.full(QH_PER_YEAR, afrr_optimization_method, dtype=object)),
            "afrr_method_note": final_result.get("afrr_method_note", np.full(QH_PER_YEAR, "", dtype=object)),
            "afrr_block_id_4h": final_result.get("afrr_block_id_4h", np.arange(QH_PER_YEAR) // 16),
            "afrr_capacity_up_price_eur_per_mw_h": afrr_capacity_up_price_h_raw if afrr_capacity_up_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_price_eur_per_mw_h": afrr_capacity_down_price_h_raw if afrr_capacity_down_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_eligible": final_result.get("afrr_capacity_eligible_h", np.zeros(QH_PER_YEAR, dtype=int)),
            "afrr_capacity_selected_market": final_result["afrr_capacity_selected_market_h"] if "afrr_capacity_selected_market_h" in final_result else np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_capacity_up_awarded": final_result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_capacity_down_awarded": final_result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_certified_capacity_up_mw": final_result["afrr_certified_capacity_up_mw_h"] if "afrr_certified_capacity_up_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_certified_capacity_down_mw": final_result["afrr_certified_capacity_down_mw_h"] if "afrr_certified_capacity_down_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_revenue_eur": final_result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_revenue_eur": final_result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_total_revenue_eur": final_result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "battery_blocked_by_afrr_capacity": final_result["battery_blocked_by_afrr_capacity"] if "battery_blocked_by_afrr_capacity" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "wholesale_opportunity_value_eur": final_result.get("wholesale_opportunity_value_eur", np.zeros(QH_PER_YEAR)),
            "wholesale_expected_value_after_capture_rate_eur": final_result.get("wholesale_expected_value_after_capture_rate_eur", np.zeros(QH_PER_YEAR)),
            "raw_up_capacity_revenue_eur": final_result.get("raw_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_capacity_revenue_eur": final_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "raw_down_capacity_revenue_eur": final_result.get("raw_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_down_capacity_revenue_eur": final_result.get("expected_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_activated_mwh": final_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)),
            "expected_down_activated_mwh": final_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_energy_expected_value_eur": final_result.get("afrr_up_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_energy_expected_value_eur": final_result.get("afrr_down_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_up_total_expected_value_eur": final_result.get("afrr_up_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_total_expected_value_eur": final_result.get("afrr_down_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "selected_market": final_result.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_capacity_direction": final_result.get("selected_capacity_direction", np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_capacity_success_rate_pct": final_result.get("afrr_capacity_success_rate_pct", np.zeros(QH_PER_YEAR)),
            "afrr_up_activation_pct": final_result.get("afrr_up_activation_pct", np.zeros(QH_PER_YEAR)),
            "afrr_down_activation_pct": final_result.get("afrr_down_activation_pct", np.zeros(QH_PER_YEAR)),
            "available_export_headroom_mwh": final_result.get("available_export_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_soc_headroom_mwh": final_result.get("available_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_discharge_from_soc_mwh": final_result.get("available_discharge_from_soc_mwh", np.zeros(QH_PER_YEAR)),
            "required_up_soc_reserve_mwh": final_result.get("required_up_soc_reserve_mwh", np.zeros(QH_PER_YEAR)),
            "required_down_soc_headroom_mwh": final_result.get("required_down_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "expected_degradation_cost_eur": final_result.get("expected_degradation_cost_eur", np.zeros(QH_PER_YEAR)),
            "future_best_market_value_eur_per_mwh": final_result.get("future_best_market_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_best_market_type": final_result.get("future_best_market_type", np.full(QH_PER_YEAR, "none", dtype=object)),
            "cross_market_spread_eur_per_mwh": final_result.get("cross_market_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "required_min_spread_eur_per_mwh": final_result.get("required_min_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "spread_condition_respected": final_result.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
            "charge_reason": final_result.get("charge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "discharge_reason": final_result.get("discharge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "stored_energy_cost_eur_per_mwh": final_result.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "effective_discharge_value_eur_per_mwh": final_result.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_expected_afrr_up_value_eur": final_result.get("future_expected_afrr_up_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_wholesale_value_eur": final_result.get("future_expected_wholesale_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_best_discharge_market": final_result.get("future_expected_best_discharge_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "wholesale_charge_for_future_afrr_flag": final_result.get("wholesale_charge_for_future_afrr_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_wholesale_flag": final_result.get("afrr_down_charge_for_future_wholesale_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_afrr_up_flag": final_result.get("afrr_down_charge_for_future_afrr_up_flag", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_spread_ok": final_result.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_spread_ok": final_result.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "forward_horizon_hours": final_result.get("forward_horizon_hours", np.zeros(QH_PER_YEAR)),
            "future_opportunity_selected": final_result.get("future_opportunity_selected", np.zeros(QH_PER_YEAR)),
            "forward_soc_before_capacity_selection_mwh": final_result.get("forward_soc_before_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "forward_soc_after_capacity_selection_mwh": final_result.get("forward_soc_after_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_soc_feasible": final_result.get("afrr_up_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_down_soc_feasible": final_result.get("afrr_down_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_soc": final_result.get("afrr_up_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_soc": final_result.get("afrr_down_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_final_combined_soc": final_result.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_final_combined_soc": final_result.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "afrr_down_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
        })

        afrr_method_summary_df = pd.DataFrame([
            {
                "aFRR Optimization Method": afrr_optimization_method,
                "Capacity start hour": afrr_capacity_start_hour,
                "Capacity end hour": afrr_capacity_end_hour,
                "Energy start hour": afrr_night_start_hour,
                "Energy end hour": afrr_night_end_hour,
                "BESS Duration (h)": bess_duration_h,
                "BESS Usable Capacity (MWh)": batt_energy_mwh,
                "BESS Gross Capacity (MWh)": bess_gross_capacity_mwh,
            }
        ])
        afrr_block_selection_df = afrr_capacity_df[[c for c in [
            "datetime",
            "afrr_optimization_method",
            "afrr_block_id_4h",
            "afrr_capacity_eligible",
            "afrr_capacity_selected_market",
            "afrr_capacity_up_awarded",
            "afrr_capacity_down_awarded",
            "afrr_capacity_total_revenue_eur",
            "selected_market",
            "selected_capacity_direction",
            "afrr_afry_block_best_market",
            "afrr_afry_rejection_reason",
            "afrr_milp_block_status",
            "afrr_milp_rejection_reason",
        ] if c in afrr_capacity_df.columns]].copy()

        excel_bytes = None
        if generate_excel_export:
            with st.spinner("Création du fichier Excel complet..."):
                inputs_df = build_inputs_dataframe(gross_inputs)
                excel_bytes = to_excel_bytes(
                    inputs_df=inputs_df,
                    summary_df=summary_df,
                    monthly_df=monthly_df,
                    hourly_df=hourly_df,
                    afrr_qh_df=afrr_qh_df,
                    afrr_daily_log_df=afrr_result["afrr_daily_log"] if afrr_result is not None else None,
                    afrr_capacity_df=afrr_capacity_df if enable_afrr_capacity else None,
                    bess_degradation_df=bess_degradation_df,
                    spain_fee_tax_df=spain_fee_tax_df,
                    spain_fee_tax_summary_df=spain_fee_tax_summary_df,
                    afrr_method_summary_df=afrr_method_summary_df,
                    afrr_block_selection_df=afrr_block_selection_df,
                    afrr_afry_heuristic_audit_df=afrr_capacity_df if afrr_optimization_method == "AFRY-style heuristic" else None,
                    afrr_milp_audit_df=afrr_capacity_df if afrr_optimization_method == "Full MILP optimization" else None,
                )

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < 60:
            optimization_time_str = f"{elapsed_time:.2f} seconds"
        else:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            optimization_time_str = f"{minutes} min {seconds} sec"

        st.subheader("Optimization Time")
        st.write(optimization_time_str)

        st.success("Simulation terminée.")

        k1, k2, k3, k4 = st.columns(4)
        if "total_revenue_including_afrr_capacity_eur" in final_result:
            total_revenue_display = final_result["total_revenue_including_afrr_capacity_eur"][0]
        elif "total_revenue_including_afrr_eur" in final_result:
            total_revenue_display = final_result["total_revenue_including_afrr_eur"][0]
        else:
            total_revenue_display = final_result["total_revenue"][0]
        total_energy_display = final_result["energy_sold_total_mwh"][0] + (np.sum(final_result["afrr_discharge_hourly_mwh"]) if "afrr_discharge_hourly_mwh" in final_result else 0.0)

        k1.metric("Revenu total", f"{total_revenue_display:,.0f} EUR")
        k2.metric("Énergie totale vendue", f"{total_energy_display:,.0f} MWh")
        k3.metric("Énergie shiftée", f"{final_result['energy_shifted_mwh'][0]:,.0f} MWh")
        k4.metric("Cycles équivalents", f"{final_result['equivalent_cycles'][0]:,.1f}")

        st.subheader("BESS availability")
        b1, b2, b3 = st.columns(3)
        b1.metric("Nominal BESS Energy Capacity", f"{batt_energy_mwh:,.2f} MWh")
        b2.metric("BESS Availability", f"{bess_availability_pct:,.1f} %")
        b3.metric("Effective BESS Energy Capacity", f"{effective_batt_energy_mwh:,.2f} MWh")

        st.subheader("Spain fees, taxes and net revenue")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Gross revenue before fees", f"{float(final_result['gross_revenue_before_fees_eur'][0]):,.0f} EUR")
        s2.metric("Variable fees & taxes", f"{float(final_result['total_variable_fees_and_taxes_eur'][0]):,.0f} EUR")
        s3.metric("Net after variable fees", f"{float(final_result['net_revenue_after_variable_fees_eur'][0]):,.0f} EUR")
        s4.metric("Cash flow after tax", f"{float(final_result['cash_flow_after_tax_and_withholding_eur'][0]):,.0f} EUR")
        s5, s6, s7, s8, s9 = st.columns(5)
        s5.metric("Effective PV capture", f"{float(np.average(sim_inputs.pv_price, weights=np.maximum(final_result['pv_direct'], 0))) if np.sum(final_result['pv_direct']) > 1e-9 else 0.0:,.2f} €/MWh")
        s6.metric("Effective DA discharge", f"{float(np.average(sim_inputs.batt_sell_price, weights=np.maximum(final_result['discharge'], 0))) if np.sum(final_result['discharge']) > 1e-9 else 0.0:,.2f} €/MWh")
        s7.metric("Effective DA charging", f"{float(np.average(sim_inputs.grid_buy_price, weights=np.maximum(final_result['grid_charge'], 0))) if np.sum(final_result['grid_charge']) > 1e-9 else 0.0:,.2f} €/MWh")
        s8.metric("Effective aFRR UP energy", f"{float(np.average(sim_inputs.afrr_discharge_price_qh, weights=np.maximum(reconciliation['afrr_discharge_qh_mwh'], 0))) if (reconciliation is not None and sim_inputs.afrr_discharge_price_qh is not None and np.sum(reconciliation['afrr_discharge_qh_mwh']) > 1e-9) else 0.0:,.2f} €/MWh")
        s9.metric("Effective aFRR DOWN charging", f"{float(np.average(sim_inputs.afrr_charge_price_qh, weights=np.maximum(reconciliation['afrr_charge_qh_mwh'], 0))) if (reconciliation is not None and sim_inputs.afrr_charge_price_qh is not None and np.sum(reconciliation['afrr_charge_qh_mwh']) > 1e-9) else 0.0:,.2f} €/MWh")

        st.subheader("Synthèse")
        summary_display_df = format_synthese_table_for_display(summary_df)
        summary_display_styler = summary_display_df.style.set_properties(
            subset=["Valeur"],
            **{"text-align": "right"}
        )
        st.dataframe(summary_display_styler, use_container_width=True, hide_index=True)

        debug = hourly_df[
            (hourly_df["datetime"] >= pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")) &
            (hourly_df["datetime"] < pd.Timestamp(f"{DEFAULT_YEAR}-06-04 00:00:00"))
        ].copy()

        st.subheader("Debug curtailment (3 premiers jours de juin)")
        st.dataframe(
            debug[[
                "datetime",
                "base_pv_generation_mwh",
                "pv_after_tso_dso_curtailment_mwh",
                "pv_after_self_curtailment_mwh",
                "pv_curtailment_candidate_mwh",
                "pv_curtailed_to_battery_mwh",
                "pv_curtailed_residual_lost_mwh",
                "pv_price_raw_eur_per_mwh",
                "pv_price_effective_eur_per_mwh",
                "self_curtailment_flag",
                "self_curtailment_reason",
                "pv_commercial_structure",
            ]],
            use_container_width=True,
        )

        st.subheader("Debug batterie - 5 premiers jours de juin")

        battery_debug = hourly_df[
            (hourly_df["datetime"] >= pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")) &
            (hourly_df["datetime"] < pd.Timestamp(f"{DEFAULT_YEAR}-06-06 00:00:00"))
        ].copy()

        battery_debug["total_battery_charge_mwh"] = (
            battery_debug["pv_to_battery_mwh"]
            + battery_debug["grid_charge_mwh"]
            + battery_debug["afrr_charge_mwh"]
            + battery_debug["pv_curtailed_to_battery_mwh"]
        )

        battery_debug["spread_check"] = (
            battery_debug["battery_sell_price_effective_eur_per_mwh"]
            - battery_debug["grid_buy_price_effective_eur_per_mwh"]
        )

        battery_debug["total_battery_discharge_mwh"] = (
            battery_debug["battery_discharge_mwh"]
            + battery_debug["afrr_discharge_mwh"]
        )

        battery_debug["wholesale_charge_price_eur_per_mwh"] = np.where(
            battery_debug["grid_charge_mwh"] > 1e-9,
            battery_debug["grid_buy_price_effective_eur_per_mwh"],
            np.where(
                battery_debug["pv_to_battery_mwh"] > 1e-9,
                battery_debug["pv_price_effective_eur_per_mwh"],
                np.nan,
            )
        )

        battery_debug["wholesale_discharge_price_eur_per_mwh"] = np.where(
            battery_debug["battery_discharge_mwh"] > 1e-9,
            battery_debug["battery_sell_price_effective_eur_per_mwh"],
            np.nan,
        )

        battery_debug["battery_activity"] = np.select(
            [
                battery_debug["total_battery_charge_mwh"] > 1e-9,
                battery_debug["total_battery_discharge_mwh"] > 1e-9,
            ],
            [
                "Charging",
                "Discharging",
            ],
            default="Idle",
        )
        # aFRR Capacity awarded MW and winning prices
        battery_debug["afrr_capacity_up_won_mw"] = np.where(
            battery_debug["afrr_capacity_up_awarded"] == 1,
            battery_debug["afrr_certified_capacity_up_mw"],
            0.0,
        )
        
        battery_debug["afrr_capacity_down_won_mw"] = np.where(
            battery_debug["afrr_capacity_down_awarded"] == 1,
            battery_debug["afrr_certified_capacity_down_mw"],
            0.0,
        )
        
        battery_debug["afrr_capacity_winning_price_eur_per_mw_h"] = np.select(
            [
                battery_debug["afrr_capacity_up_awarded"] == 1,
                battery_debug["afrr_capacity_down_awarded"] == 1,
            ],
            [
                battery_debug["afrr_capacity_up_price_eur_per_mw_h"],
                battery_debug["afrr_capacity_down_price_eur_per_mw_h"],
            ],
            default=np.nan,
        )
        
        battery_debug["afrr_capacity_winning_direction"] = np.select(
            [
                battery_debug["afrr_capacity_up_awarded"] == 1,
                battery_debug["afrr_capacity_down_awarded"] == 1,
            ],
            [
                "UP",
                "DOWN",
            ],
            default="None",
        )
        st.dataframe(
            battery_debug[[
                "datetime",
                "battery_activity",
                "battery_soc_mwh_end",
                "total_battery_charge_mwh",
                "total_battery_discharge_mwh",
                "pv_to_battery_mwh",
                "pv_curtailed_to_battery_mwh",
                "grid_charge_mwh",
                "afrr_charge_mwh",
                "battery_discharge_mwh",
                "afrr_discharge_mwh",
                "afrr_capacity_winning_direction",
                "afrr_capacity_up_won_mw",
                "afrr_capacity_down_won_mw",
                "afrr_capacity_winning_price_eur_per_mw_h",
                "wholesale_charge_price_eur_per_mwh",
                "avg_stored_charge_price_eur_per_mwh",
                "required_discharge_price_eur_per_mwh",
                "wholesale_discharge_price_eur_per_mwh",
                "battery_sale_revenue_eur",
                "grid_charge_cost_eur",
                "wholesale_cycle_cost_eur",
                "afrr_charge_cost_eur",
                "afrr_sale_revenue_eur",
                "afrr_cycle_cost_eur",
                "afrr_net_revenue_eur",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(10, 4.8))

            fee_total = float(np.sum(spain_fee_tax_breakdown["total_variable_fees_and_taxes_eur"]))
            bars = [
                float(np.sum(spain_fee_tax_breakdown["pv_revenue_gross_eur"])) / 1e6,
                float(np.sum(spain_fee_tax_breakdown["da_discharge_revenue_gross_eur"])) / 1e6,
                -float(np.sum(spain_fee_tax_breakdown["da_charge_cost_gross_eur"])) / 1e6,
                float(np.sum(spain_fee_tax_breakdown["afrr_energy_revenue_gross_eur"])) / 1e6,
                -float(np.sum(spain_fee_tax_breakdown["afrr_charge_cost_gross_eur"])) / 1e6,
                float(np.sum(spain_fee_tax_breakdown["afrr_capacity_revenue_gross_eur"])) / 1e6,
                -fee_total / 1e6,
                float(np.sum(spain_fee_tax_breakdown["net_revenue_after_variable_fees_eur"])) / 1e6,
                float(pure_pv_benchmark["total_pv_only_revenue_eur"][0]) / 1e6,
            ]

            labels = [
                "PV direct gross",
                "DA gross",
                "DA Charge Cost",
                "aFRR Energy gross",
                "aFRR Charge Cost",
                "aFRR Capacity gross",
                "Variable fees & taxes",
                "Net after variable fees",
                "PV-only",
            ]
            colors = [
                "orange",
                "tab:blue",
                "tab:red",
                "tab:cyan",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:green",
                "orange",
            ]

            ax1.bar(labels, bars, color=colors)
            ax1.axhline(0, linewidth=0.8, color="black")
            ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax1.set_title("Revenue Breakdown")
            ax1.set_ylabel("million €")
            ax1.tick_params(axis="x", rotation=30)
            fig1.tight_layout()

            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            pv_vals = monthly_df["pv_revenue_keur_per_mw"].to_numpy(dtype=float)
            afrr_vals = monthly_df["afrr_net_revenue"].to_numpy(dtype=float) / max(batt_power_mw, 1e-12) / 1000.0
            afrr_capacity_vals = monthly_df["afrr_capacity_total_revenue"].to_numpy(dtype=float) / max(batt_power_mw, 1e-12) / 1000.0 if "afrr_capacity_total_revenue" in monthly_df.columns else np.zeros(len(monthly_df))
            bess_vals = monthly_df["bess_revenue_keur_per_mw"].to_numpy(dtype=float) - afrr_vals - afrr_capacity_vals

            ax2.bar(x, pv_vals, width=0.65, color="orange", label="PV")
            ax2.bar(x, bess_vals, width=0.65, bottom=pv_vals, color="lightgreen", label="DA Arbitrage")
            ax2.bar(x, afrr_capacity_vals, width=0.65, bottom=pv_vals + bess_vals, label="aFRR Capacity")
            ax2.bar(x, afrr_vals, width=0.65, bottom=pv_vals + bess_vals + afrr_capacity_vals, color="blue", label="aFRR Energy")

            ax2.set_title("Specific Monthly Revenues per MW")
            ax2.set_ylabel("k€/MW")
            ax2.set_xlabel("Month")
            ax2.set_xticks(x)
            ax2.set_xticklabels(monthly_df["month"], rotation=45)
            ax2.legend()
            fig2.tight_layout()

            st.pyplot(fig2)
            plt.close(fig2)

        def _plot_monthly_valued_energy():
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))

            ax3.plot(monthly_df["month"], monthly_df["pv_direct_mwh"], label="PV direct")
            ax3.plot(monthly_df["month"], monthly_df["shifted_mwh"], label="Énergie shiftée wholesale")
            ax3.plot(monthly_df["month"], monthly_df["pv_only_direct_mwh"], label="PV-only direct")

            if "afrr_discharge_mwh" in monthly_df.columns:
                ax3.plot(monthly_df["month"], monthly_df["afrr_discharge_mwh"], label="Décharge aFRR")

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_to_battery_mwh_actual"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_residual_lost_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed lost"
                )

            ax3.set_title("Énergies valorisées par mois")
            ax3.set_ylabel("MWh")
            ax3.set_xlabel("Mois")
            ax3.legend()
            ax3.tick_params(axis="x", rotation=45)
            fig3.tight_layout()
            return fig3

        def _plot_specific_monthly_revenues_per_mwh():
            fig5, ax5 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.34

            pv_vals_mwh = monthly_df["pv_revenue_eur_per_mwh"].to_numpy(dtype=float)
            bess_vals_mwh = monthly_df["bess_revenue_eur_per_mwh"].to_numpy(dtype=float)
            pv_only_vals_mwh = (
                monthly_df["pv_only_revenue"].to_numpy(dtype=float)
                / monthly_df["pv_only_direct_mwh"].clip(lower=1e-12).to_numpy(dtype=float)
            )

            ax5.bar(
                x - width / 2,
                pv_vals_mwh,
                width=width,
                color="orange",
                label="PV hybride"
            )

            ax5.bar(
                x + width / 2,
                bess_vals_mwh,
                width=width,
                color="green",
                label="BESS"
            )

            ax5.plot(
                x,
                pv_only_vals_mwh,
                marker="o",
                linewidth=2.0,
                label="PV-only Project"
            )

            ax5.set_title("Specific Monthly Revenues per MWh")
            ax5.set_ylabel("€/MWh")
            ax5.set_xlabel("Month")
            ax5.set_xticks(x)
            ax5.set_xticklabels(monthly_df["month"], rotation=45)
            ax5.legend()
            fig5.tight_layout()
            return fig5

        def _plot_dispatch_period(start_date, end_date, title, figsize=(12, 5), day_locator=False):
            df_plot = hourly_df[
                (hourly_df["datetime"] >= start_date) &
                (hourly_df["datetime"] < end_date)
            ].copy()

            fig, ax1 = plt.subplots(figsize=figsize)
            bar_width = 0.03

            ax1.fill_between(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                alpha=0.5,
                label="PV → Réseau"
            )
            ax1.plot(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                linewidth=1.8
            )

            ax1.bar(
                df_plot["datetime"],
                df_plot["battery_discharge_mwh"],
                width=bar_width,
                label="Batterie → Réseau (wholesale)",
                alpha=0.8,
                color="green"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["pv_to_battery_mwh"],
                width=bar_width,
                label="PV → Batterie",
                alpha=0.6,
                color="red"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["grid_charge_mwh"],
                width=bar_width,
                bottom=-df_plot["pv_to_battery_mwh"],
                label="Réseau → Batterie",
                alpha=0.6
            )

            if "afrr_discharge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    df_plot["afrr_discharge_mwh"],
                    width=bar_width,
                    label="aFRR → Décharge",
                    alpha=0.5,
                    color="purple"
                )

            if "afrr_charge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["afrr_charge_mwh"],
                    width=bar_width,
                    label="aFRR → Charge",
                    alpha=0.5,
                    color="blue"
                )

            if "pv_curtailment_candidate_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    linewidth=1.5,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_curtailed_to_battery_mwh"],
                    width=bar_width,
                    label="PV curtailed → battery",
                    alpha=0.6
                )

            if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailed_residual_lost_mwh"],
                    linestyle=":",
                    linewidth=1.8,
                    label="PV curtailed lost"
                )

            ax1.axhline(0, linewidth=1)
            ax1.set_ylabel("Flux énergie (MWh)")
            ax1.set_xlabel("Heure")
            if day_locator:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                ax1.tick_params(axis="x", rotation=45)
            else:
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
                ax1.tick_params(axis="x", rotation=45)

            ax2 = ax1.twinx()
            ax2.plot(
                df_plot["datetime"],
                df_plot["pv_price_effective_eur_per_mwh"],
                linestyle="--",
                alpha=0.7,
                label="Prix spot PV effectif"
            )
            ax2.set_ylabel("Prix (EUR/MWh)")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
            )
            ax1.set_title(title)
            fig.tight_layout(rect=[0, 0.12, 1, 1])
            return fig

        def _plot_full_june_dispatch():
            june_blocks = [
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-11 00:00:00"),
                    "1-10 juin",
                ),
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-11 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-21 00:00:00"),
                    "11-20 juin",
                ),
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-21 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-07-01 00:00:00"),
                    "21-30 juin",
                ),
            ]

            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(16, 12),
                sharey=True,
            )

            bar_width = 0.03
            legend_handles = []
            legend_labels = []

            for ax1, (start_date, end_date, block_title) in zip(axes, june_blocks):
                df_plot = hourly_df[
                    (hourly_df["datetime"] >= start_date) &
                    (hourly_df["datetime"] < end_date)
                ].copy()

                ax1.fill_between(
                    df_plot["datetime"],
                    df_plot["pv_direct_mwh"],
                    color="orange",
                    alpha=0.5,
                    label="PV → Réseau"
                )
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_direct_mwh"],
                    color="orange",
                    linewidth=1.8
                )

                ax1.bar(
                    df_plot["datetime"],
                    df_plot["battery_discharge_mwh"],
                    width=bar_width,
                    label="Batterie → Réseau (wholesale)",
                    alpha=0.8,
                    color="green"
                )

                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_to_battery_mwh"],
                    width=bar_width,
                    label="PV → Batterie",
                    alpha=0.6,
                    color="red"
                )

                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["grid_charge_mwh"],
                    width=bar_width,
                    bottom=-df_plot["pv_to_battery_mwh"],
                    label="Réseau → Batterie",
                    alpha=0.6
                )

                if "afrr_discharge_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        df_plot["afrr_discharge_mwh"],
                        width=bar_width,
                        label="aFRR → Décharge",
                        alpha=0.5,
                        color="purple"
                    )

                if "afrr_charge_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        -df_plot["afrr_charge_mwh"],
                        width=bar_width,
                        label="aFRR → Charge",
                        alpha=0.5,
                        color="blue"
                    )

                if "pv_curtailment_candidate_mwh" in df_plot.columns:
                    ax1.plot(
                        df_plot["datetime"],
                        df_plot["pv_curtailment_candidate_mwh"],
                        linestyle="--",
                        linewidth=1.5,
                        label="PV curtailed"
                    )

                if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        -df_plot["pv_curtailed_to_battery_mwh"],
                        width=bar_width,
                        label="PV curtailed → battery",
                        alpha=0.6
                    )

                if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                    ax1.plot(
                        df_plot["datetime"],
                        df_plot["pv_curtailed_residual_lost_mwh"],
                        linestyle=":",
                        linewidth=1.8,
                        label="PV curtailed lost"
                    )

                ax1.axhline(0, linewidth=1)
                ax1.set_ylabel("Flux énergie (MWh)")
                ax1.set_title(f"Dispatch énergétique - {block_title}")
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                ax1.tick_params(axis="x", rotation=45)

                ax2 = ax1.twinx()
                ax2.plot(
                    df_plot["datetime"],
                    df_plot["pv_price_effective_eur_per_mwh"],
                    linestyle="--",
                    alpha=0.7,
                    label="Prix spot PV effectif"
                )
                ax2.set_ylabel("Prix (EUR/MWh)")

                if not legend_handles:
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    legend_handles = lines_1 + lines_2
                    legend_labels = labels_1 + labels_2

            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=4,
                frameon=False,
            )

            fig.suptitle("Dispatch énergétique - mois complet de juin", fontsize=14)
            fig.tight_layout(rect=[0, 0.08, 1, 0.96])
            return fig

        def _plot_curtailment_monthly():
            fig8, ax8 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.26

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax8.bar(
                    x - width,
                    monthly_df["pv_curtailment_candidate_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax8.bar(
                    x,
                    monthly_df["pv_curtailed_to_battery_mwh_actual"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax8.bar(
                    x + width,
                    monthly_df["pv_curtailed_residual_lost_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed lost"
                )

            ax8.set_title("Curtailment mensuel PV")
            ax8.set_ylabel("MWh")
            ax8.set_xlabel("Mois")
            ax8.set_xticks(x)
            ax8.set_xticklabels(monthly_df["month"], rotation=45)
            ax8.legend()
            fig8.tight_layout()
            return fig8

        c3, c4 = st.columns(2)
        with c3:
            fig3 = _plot_monthly_valued_energy()
            st.pyplot(fig3)
            plt.close(fig3)
        with c4:
            fig5 = _plot_specific_monthly_revenues_per_mwh()
            st.pyplot(fig5)
            plt.close(fig5)

        st.subheader("Dispatch énergétique")

        # Layout requested by the user:
        # - Left side: the two short-period dispatch charts stacked vertically,
        #   each on its own row.
        # - Right side: the full-June dispatch chart spanning the same overall
        #   vertical space as the two left charts combined.
        dispatch_left_col, dispatch_right_col = st.columns([1, 1])

        with dispatch_left_col:
            fig = _plot_dispatch_period(
                pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00"),
                pd.Timestamp(f"{DEFAULT_YEAR}-06-06 00:00:00"),
                "Dispatch énergétique - 5 premiers jours de juin",
                figsize=(12, 5.8),
                day_locator=False,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            fig = _plot_dispatch_period(
                pd.Timestamp(f"{DEFAULT_YEAR}-05-27 00:00:00"),
                pd.Timestamp(f"{DEFAULT_YEAR}-06-06 00:00:00"),
                "Dispatch énergétique - 5 derniers jours de mai + 5 premiers jours de juin",
                figsize=(12, 5.8),
                day_locator=True,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with dispatch_right_col:
            fig = _plot_full_june_dispatch()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c7, c8 = st.columns(2)
        with c7:
            st.subheader("Comparaison Revenu PV-only vs Hybrid")

            fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))

            pv_only_monthly_keur = monthly_df["pv_only_revenue"].to_numpy(dtype=float) / 1000.0
            hybrid_monthly_keur = monthly_df["net_revenue"].to_numpy(dtype=float) / 1000.0

            ax_cmp.plot(
                x,
                pv_only_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="PV-only"
            )

            ax_cmp.plot(
                x,
                hybrid_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="Hybrid (PV + BESS)"
            )

            if enable_cfd and "pv_only_cfd_revenue" in monthly_df.columns:
                pv_only_cfd_monthly_keur = (
                    monthly_df["pv_only_cfd_revenue"].to_numpy(dtype=float) / 1000.0
                )

                ax_cmp.plot(
                    x,
                    pv_only_cfd_monthly_keur,
                    marker="o",
                    linewidth=2.0,
                    label="PV-only-CfD"
                )

            ax_cmp.set_title("Comparaison Revenu PV-only vs Hybrid")
            ax_cmp.set_ylabel("kEUR")
            ax_cmp.set_xlabel("Mois")
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(monthly_df["month"], rotation=45)
            ax_cmp.legend()
            fig_cmp.tight_layout()

            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

        with c8:
            st.subheader("Curtailment Specifics")
            fig8 = _plot_curtailment_monthly()
            st.pyplot(fig8)
            plt.close(fig8)

        st.subheader("Table mensuelle")
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        st.subheader("Exports")
        if generate_excel_export and excel_bytes is not None:
            st.download_button(
                "Télécharger cette simulation complète (Excel)",
                data=excel_bytes,
                file_name="simulation_complete_hybride_pv_bess.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click="ignore",
            )
        else:
            st.info(
                "Export Excel complet désactivé pour cette simulation. "
                "Cochez l'option avant de lancer la simulation si vous voulez générer le fichier Excel complet."
            )
        st.download_button(
            "Télécharger l'horaire en CSV",
            data=hourly_df.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_horaire_hybride.csv",
            mime="text/csv",
            on_click="ignore",
        )

        if afrr_qh_df is not None:
            st.download_button(
                "Télécharger l'aFRR quart-horaire en CSV",
                data=afrr_qh_df.to_csv(index=False).encode("utf-8"),
                file_name="dispatch_afrr_quart_horaire.csv",
                mime="text/csv",
                on_click="ignore",
            )

    except Exception as e:
        st.error(f"Erreur: {e}")
