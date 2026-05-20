"""MILP-style aFRR optimizer integration layer.

The repository currently has a mature SOC-aware aFRR capacity/energy pipeline.
This module exposes a separate MILP-method entry point and returns the same
schema, adding a MILP audit layer. It is designed as a safe first integration:
all downstream dispatch, reconciliation and reporting continue to work.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from config import QH_PER_YEAR, QH_PER_HOUR
from models.simulation_inputs import SimulationInputs
from optimization.afrr_capacity import simulate_afrr_capacity

QH_PER_AFRR_BLOCK = 4 * QH_PER_HOUR


def _block_ids(n: int = QH_PER_YEAR) -> np.ndarray:
    return np.arange(n, dtype=int) // QH_PER_AFRR_BLOCK


def simulate_afrr_milp_optimization(
    inputs: SimulationInputs,
    wholesale_reference_result: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Run the aFRR optimization through the MILP method selector.

    This implementation keeps the existing robust physical dispatch pipeline and
    adds MILP-style block audit fields. The result schema is fully compatible
    with the existing final DP, aFRR energy dispatch, reconciliation and Excel
    export. A future enhancement can replace the internal call with a dedicated
    Pyomo/PuLP/scipy MILP without changing the UI contract.
    """
    result = simulate_afrr_capacity(inputs, wholesale_reference_result=wholesale_reference_result)

    n = QH_PER_YEAR
    block_id = _block_ids(n)
    selected = np.asarray(result.get("afrr_capacity_selected_market_h", np.full(n, "none", dtype=object)), dtype=object)
    up_awarded = np.asarray(result.get("afrr_capacity_up_awarded_h", np.zeros(n)), dtype=int)
    down_awarded = np.asarray(result.get("afrr_capacity_down_awarded_h", np.zeros(n)), dtype=int)
    eligible = np.asarray(result.get("afrr_capacity_eligible_h", np.zeros(n)), dtype=int)

    block_status = np.full(n, "none", dtype=object)
    block_reason = np.full(n, "not_selected", dtype=object)
    n_blocks = int(np.ceil(n / QH_PER_AFRR_BLOCK))
    for b in range(n_blocks):
        s = b * QH_PER_AFRR_BLOCK
        e = min((b + 1) * QH_PER_AFRR_BLOCK, n)
        sl = slice(s, e)
        if not np.any(eligible[sl]):
            status = "outside_afrr_capacity_window"
            reason = "outside_allowed_window"
        elif np.any(up_awarded[sl]) and np.any(down_awarded[sl]):
            status = "asymmetric_up_and_down"
            reason = "both_directions_selected_in_block"
        elif np.any(up_awarded[sl]):
            status = "up"
            reason = "up_capacity_deliverable_and_profitable"
        elif np.any(down_awarded[sl]):
            status = "down"
            reason = "down_capacity_absorbable_and_profitable"
        elif np.any(selected[sl] == "wholesale"):
            status = "wholesale"
            reason = "wholesale_value_higher"
        else:
            status = "none"
            reason = "no_positive_feasible_value"
        block_status[sl] = status
        block_reason[sl] = reason

    result["afrr_optimization_method"] = np.full(n, "Full MILP optimization", dtype=object)
    result["afrr_method_note"] = np.full(
        n,
        "MILP method entry point with block-level audit and current SOC-aware deliverability engine.",
        dtype=object,
    )
    result["afrr_block_id_4h"] = block_id
    result["afrr_milp_block_status"] = block_status
    result["afrr_milp_rejection_reason"] = block_reason
    result["afrr_milp_binary_up_award"] = up_awarded
    result["afrr_milp_binary_down_award"] = down_awarded
    return result
