from dataclasses import dataclass
import numpy as np
from config import PV_ZERO_TOLERANCE_MWH

@dataclass
class SimulationInputs:
    batt_power_mw: float
    batt_energy_mwh: float
    pv_dc_mw: float
    productible_kwh_per_kwp: float
    pv_losses_pct: float
    plant_availability_pct: float
    eta_charge: float
    eta_discharge: float

    # Effective prices used by the optimizer/economics
    pv_price: np.ndarray
    batt_sell_price: np.ndarray
    grid_buy_price: np.ndarray

    # PV available for direct sale / standard PV-to-battery charging
    solar_profile: np.ndarray

    # PV curtailed but optionally recoverable into battery only
    curtailed_pv_recoverable_mwh: np.ndarray | None = None

    # BESS availability reporting: batt_energy_mwh is the effective usable capacity used by the model.
    nominal_batt_energy_mwh: float = 0.0
    bess_availability_pct: float = 100.0
    bess_duration_h: float = 4.0
    gross_batt_energy_mwh: float = 0.0
    technical_eta_charge: float = 1.0
    technical_eta_discharge: float = 1.0

    nightly_bess_revenue_eur: float = 0.0
    soc_steps: int = 101
    initial_soc_mwh: float = 0.0
    final_soc_mwh: float = 0.0
    min_soc_pct: float = 0.0
    max_soc_pct: float = 100.0
    grid_export_limit_mw: float = 0.0
    cycle_cost_eur_per_mwh: float = 0.0
    charge_quantile: float = 100.0
    discharge_quantile: float = 0.0
    max_cycles_per_year: float = 1.0
    min_spread_arbitrage_eur_per_mwh: float = 0.0
    # Forward-looking cross-market optimization controls
    forward_optimization_horizon_hours: float = 24.0
    afrr_up_cross_market_min_spread_eur_per_mwh: float = 20.0
    afrr_down_to_wholesale_min_spread_eur_per_mwh: float = 20.0

    # Capture rates
    pv_capture_rate_pct: float = 100.0
    bess_capture_rate_pct: float = 100.0

    # Spain-specific taxes, grid fees and market fees.
    # Marginal fees are used to build dispatch-effective prices.
    # Financial-only taxes are kept for reporting and must not affect dispatch.
    grid_import_fee_eur_per_mwh: float = 0.0
    grid_export_fee_eur_per_mwh: float = 0.0
    omie_buy_fee_eur_per_mwh: float = 0.0
    omie_sell_fee_eur_per_mwh: float = 0.0
    ree_system_fee_eur_per_mwh: float = 0.0
    imbalance_cost_pv_eur_per_mwh: float = 0.0
    imbalance_cost_bess_eur_per_mwh: float = 0.0
    afrr_capacity_fee_pct: float = 0.0
    afrr_energy_fee_pct: float = 0.0
    afrr_energy_fee_eur_per_mwh: float = 0.0
    ivpee_generation_tax_pct: float = 0.0
    apply_ivpee_to_pv: bool = True
    apply_ivpee_to_bess_export: bool = False
    apply_ivpee_to_afrr_energy: bool = False
    apply_ivpee_to_afrr_capacity: bool = False
    corporate_tax_pct: float = 0.0
    withholding_tax_pct: float = 0.0
    local_fixed_tax_eur_per_year: float = 0.0

    # aFRR inputs
    afrr_optimization_method: str = "Current model"
    enable_afrr: bool = False
    afrr_charge_price_qh: np.ndarray | None = None
    afrr_discharge_price_qh: np.ndarray | None = None
    afrr_min_spread_eur_per_mwh: float = 0.0
    afrr_cycle_cost_eur_per_mwh: float = 0.0
    afrr_max_events_per_day: int = 1
    # aFRR Energy eligibility window. Keeps legacy field names for compatibility.
    afrr_night_start_hour: int = 20
    afrr_night_end_hour: int = 8
    afrr_pv_zero_tolerance_mwh: float = PV_ZERO_TOLERANCE_MWH
    afrr_n_qh_per_side: int = 4
    afrr_energy_down_activation_pct: float = 100.0
    afrr_energy_up_activation_pct: float = 100.0

    # aFRR Capacity inputs
    enable_afrr_capacity: bool = False
    afrr_capacity_up_price_h: np.ndarray | None = None
    afrr_capacity_down_price_h: np.ndarray | None = None
    afrr_certified_capacity_pct: float = 100.0
    afrr_capacity_success_rate_pct: float = 80.0
    allow_afrr_energy_without_capacity: bool = True
    afrr_certified_capacity_up_mw: float = 0.0
    afrr_certified_capacity_down_mw: float = 0.0
    # aFRR Capacity eligibility window.
    afrr_capacity_start_hour: int = 20
    afrr_capacity_end_hour: int = 8
    # Internal quarter-hour market selection used to block wholesale and gate aFRR energy.
    afrr_capacity_selected_market_h: np.ndarray | None = None
    # Expected activated energy arrays from Phase-1 aFRR capacity selection.
    # These are used to keep physical aFRR energy dispatch aligned with the
    # expected MWh used in the capacity value comparison.
    afrr_expected_up_activated_mwh_qh: np.ndarray | None = None
    afrr_expected_down_activated_mwh_qh: np.ndarray | None = None

    # Curtailment
    enable_tso_dso_curtailment: bool = False
    tso_dso_monthly_curtailment_pct: np.ndarray | None = None
    enable_self_curtailment: bool = False
    curtailment_threshold_eur_per_mwh: float = -1.0
    pv_commercial_structure: str = "Fully merchant"  # Fully merchant / With CfD / With PPA
    cfd_price_eur_per_mwh: float = 0.0
    negative_price_rule: bool = False
    consecutive_negative_hours_limit: int = 6
    ppa_price_eur_per_mwh: float = 0.0
    charge_battery_if_curtailment: bool = False
    enable_cfd: bool = False
    cfd_price_standalone_eur_per_mwh: float = 0.0
    enable_ppa: bool = False
    ppa_price_standalone_eur_per_mwh: float = 0.0
    project_lifetime_years: int = 1
    bess_degradation_curve_pct: np.ndarray | None = None
    degraded_bess_energy_by_year_mwh: np.ndarray | None = None
