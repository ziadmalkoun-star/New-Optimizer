"""AFRY-style aFRR expected-value block allocation.

This module is intentionally compatible with the existing aFRR capacity result
schema. It works as a screening / expected-value allocator over 4-hour blocks,
then returns quarter-hour arrays so the existing Streamlit reporting and final
reconciliation pipeline can continue to work unchanged.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from config import QH_PER_YEAR, QH_PER_HOUR
from models.simulation_inputs import SimulationInputs
from optimization.afrr_capacity import simulate_afrr_capacity

QH_PER_AFRR_BLOCK = 4 * QH_PER_HOUR  # 4 hours x 4 quarter-hours/hour = 16


def _block_ids(n: int = QH_PER_YEAR) -> np.ndarray:
    return np.arange(n, dtype=int) // QH_PER_AFRR_BLOCK


def _add_method_audit(result: Dict[str, np.ndarray], method_name: str, note: str) -> Dict[str, np.ndarray]:
    out = dict(result)
    n = QH_PER_YEAR
    out["afrr_optimization_method"] = np.full(n, method_name, dtype=object)
    out["afrr_method_note"] = np.full(n, note, dtype=object)
    out["afrr_block_id_4h"] = _block_ids(n)
    out["afrr_block_start_qh"] = (out["afrr_block_id_4h"] * QH_PER_AFRR_BLOCK).astype(int)
    out["afrr_block_end_qh_exclusive"] = np.minimum(out["afrr_block_start_qh"] + QH_PER_AFRR_BLOCK, n).astype(int)
    return out


def simulate_afrr_afry_heuristic(
    inputs: SimulationInputs,
    wholesale_reference_result: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Run an AFRY-style 4-hour block expected-value screening allocation.

    The existing capacity optimizer already computes expected UP/DOWN values,
    success-rate adjusted capacity revenue, activation value, SOC feasibility,
    time-window eligibility, and wholesale opportunity cost. This wrapper keeps
    that physical/economic logic and adds a block-level audit layer so the user
    can inspect the method as an AFRY-style expected-value allocation.
    """
    result = simulate_afrr_capacity(inputs, wholesale_reference_result=wholesale_reference_result)

    # Add block-level winning direction for audit. We deliberately keep the
    # original quarter-hour awards to preserve current behavior and downstream
    # reconciliation, while making the block comparison explicit in Excel.
    n_blocks = int(np.ceil(QH_PER_YEAR / QH_PER_AFRR_BLOCK))
    block_best = np.full(QH_PER_YEAR, "none", dtype=object)
    block_up_value = np.zeros(QH_PER_YEAR, dtype=float)
    block_down_value = np.zeros(QH_PER_YEAR, dtype=float)
    block_wholesale_value = np.zeros(QH_PER_YEAR, dtype=float)
    block_reason = np.full(QH_PER_YEAR, "not_selected", dtype=object)

    up_val = np.asarray(result.get("afrr_up_total_expected_value_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    down_val = np.asarray(result.get("afrr_down_total_expected_value_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    wh_val = np.asarray(result.get("wholesale_expected_value_after_capture_rate_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    eligible = np.asarray(result.get("afrr_capacity_eligible_h", np.zeros(QH_PER_YEAR)), dtype=int)

    for b in range(n_blocks):
        s = b * QH_PER_AFRR_BLOCK
        e = min((b + 1) * QH_PER_AFRR_BLOCK, QH_PER_YEAR)
        sl = slice(s, e)
        b_up = float(np.nansum(up_val[sl]))
        b_down = float(np.nansum(down_val[sl]))
        b_wh = float(np.nansum(wh_val[sl]))
        block_up_value[sl] = b_up
        block_down_value[sl] = b_down
        block_wholesale_value[sl] = b_wh
        if not np.any(eligible[sl]):
            winner = "outside_afrr_capacity_window"
            reason = "outside_allowed_window"
        else:
            best = max(b_up, b_down, b_wh, 0.0)
            if best <= 0:
                winner = "none"
                reason = "no_positive_expected_value"
            elif b_up == best:
                winner = "afrr_up_capacity"
                reason = "up_expected_value_best"
            elif b_down == best:
                winner = "afrr_down_capacity"
                reason = "down_expected_value_best"
            else:
                winner = "wholesale"
                reason = "wholesale_opportunity_best"
        block_best[sl] = winner
        block_reason[sl] = reason

    result["afrr_afry_block_best_market"] = block_best
    result["afrr_afry_block_up_value_eur"] = block_up_value
    result["afrr_afry_block_down_value_eur"] = block_down_value
    result["afrr_afry_block_wholesale_value_eur"] = block_wholesale_value
    result["afrr_afry_rejection_reason"] = block_reason

    return _add_method_audit(
        result,
        "AFRY-style heuristic",
        "Expected-value 4-hour block screening using current physical deliverability checks.",
    )
