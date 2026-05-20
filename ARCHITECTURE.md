# Refactored architecture

## Proposed tree

```text
project_root/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── models/
│   ├── __init__.py
│   └── simulation_inputs.py
├── data/
│   ├── __init__.py
│   ├── loaders.py
│   ├── validators.py
│   ├── solar_profile.py
│   └── curtailment.py
├── optimization/
│   ├── __init__.py
│   ├── dispatch_optimizer.py
│   ├── afrr.py
│   ├── afrr_capacity.py
│   ├── reconciliation.py
│   └── forward_curves.py
├── finance/
│   ├── __init__.py
│   ├── pricing.py
│   ├── taxes.py
│   ├── revenues.py
│   └── benchmarks.py
├── utils/
│   ├── __init__.py
│   ├── dataframe_utils.py
│   ├── time_utils.py
│   └── math_utils.py
├── visualization/
│   ├── __init__.py
│   └── streamlit_ui.py
└── exports/
    ├── __init__.py
    └── excel_export.py
```

## Function/class mapping

| File | Moved content |
|---|---|
| `config.py` | constants, `APP_DIR`, `BUILTIN_CURTAILMENT_CURVE`, `_open_builtin_file` |
| `models/simulation_inputs.py` | `SimulationInputs` |
| `data/validators.py` | `_validate_array_length` |
| `data/loaders.py` | `_read_single_column_csv`, `_read_single_column_csv_qh`, `read_monthly_curtailment_excel`, `read_bess_degradation_excel` |
| `data/solar_profile.py` | `build_standard_france_solar_profile`, `build_pv_generation_mwh` |
| `data/curtailment.py` | `apply_tso_dso_curtailment`, `apply_self_curtailment` |
| `optimization/forward_curves.py` | `compute_forward_cross_market_value_curves` |
| `optimization/dispatch_optimizer.py` | `optimize_dispatch_dp` |
| `optimization/afrr.py` | `_select_best_daily_afrr_competing_blocks`, `simulate_afrr_night_arbitrage` |
| `optimization/afrr_capacity.py` | `simulate_afrr_capacity`, `enforce_afrr_capacity_deliverability_from_final_dispatch` |
| `optimization/reconciliation.py` | `build_combined_soc_with_afrr`, `reconcile_wholesale_afrr_dispatch_qh`, `enforce_hard_annual_cycle_cap_on_reconciliation`, `build_final_result_after_market_arbitration`, `add_afrr_capacity_to_final_result` |
| `finance/pricing.py` | `build_effective_dispatch_prices` |
| `finance/taxes.py` | `compute_spain_fee_tax_breakdown`, `summarize_spain_fee_tax_breakdown` |
| `finance/benchmarks.py` | `build_pure_pv_benchmark` |
| `finance/revenues.py` | `build_summary_table`, `format_synthese_number`, `format_synthese_table_for_display` |
| `utils/math_utils.py` | `rolling_forward_max`, `_make_flat_curve` |
| `utils/time_utils.py` | `build_quarter_hour_index` |
| `utils/dataframe_utils.py` | `_make_qh_dataframe`, `monthly_dataframe`, `build_inputs_dataframe` |
| `exports/excel_export.py` | `to_excel_bytes` |
| `visualization/streamlit_ui.py` | full Streamlit UI/app orchestration as `run_app()` |
| `app.py` | lightweight entry point only |

## Changed imports

- `app.py` now imports `run_app` from `visualization.streamlit_ui`.
- Domain modules import shared constants from `config.py`.
- Domain modules import `SimulationInputs` from `models.simulation_inputs`.
- Validation, time, math, pricing, optimization, finance, and export helpers are imported from their dedicated modules.

## Verification checklist

- [x] All files generated in a multi-file project structure.
- [x] `app.py` reduced to a lightweight Streamlit entry point.
- [x] Original function and variable names preserved where possible.
- [x] `__init__.py` files added to package directories.
- [x] Python syntax checked with `python -m compileall .`.
- [x] Import smoke-test passed using a temporary Streamlit stub because Streamlit is not installed in this execution container.
- [ ] Run `streamlit run app.py` locally after installing dependencies.
- [ ] Compare outputs against the original script using the same uploaded input files.
