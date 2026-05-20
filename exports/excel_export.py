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
from utils.dataframe_utils import _make_qh_dataframe, monthly_dataframe, build_inputs_dataframe
from finance.revenues import build_summary_table

def to_excel_bytes(
    inputs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    afrr_qh_df: pd.DataFrame | None = None,
    afrr_daily_log_df: pd.DataFrame | None = None,
    afrr_capacity_df: pd.DataFrame | None = None,
    bess_degradation_df: pd.DataFrame | None = None,
    spain_fee_tax_df: pd.DataFrame | None = None,
    spain_fee_tax_summary_df: pd.DataFrame | None = None,
    afrr_method_summary_df: pd.DataFrame | None = None,
    afrr_afry_heuristic_audit_df: pd.DataFrame | None = None,
    afrr_milp_audit_df: pd.DataFrame | None = None,
    afrr_block_selection_df: pd.DataFrame | None = None,
) -> bytes:
    def _write_excel(output_buffer: io.BytesIO, engine: str, engine_kwargs: dict | None = None) -> None:
        writer_kwargs = {"engine": engine}
        if engine_kwargs is not None:
            writer_kwargs["engine_kwargs"] = engine_kwargs
        with pd.ExcelWriter(output_buffer, **writer_kwargs) as writer:
            inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            monthly_df.to_excel(writer, sheet_name="Monthly", index=False)
            hourly_df.to_excel(writer, sheet_name="Hourly", index=False)

            if afrr_qh_df is not None:
                afrr_qh_df.to_excel(writer, sheet_name="aFRR_QH", index=False)

            if afrr_daily_log_df is not None:
                afrr_daily_log_df.to_excel(writer, sheet_name="aFRR_Daily_Log", index=False)

            if afrr_capacity_df is not None:
                afrr_capacity_df.to_excel(writer, sheet_name="aFRR_Capacity", index=False)

            if bess_degradation_df is not None:
                bess_degradation_df.to_excel(writer, sheet_name="BESS_Degradation", index=False)

            if spain_fee_tax_df is not None:
                spain_fee_tax_df.to_excel(writer, sheet_name="Spain Fees and Taxes", index=False)

            if spain_fee_tax_summary_df is not None:
                spain_fee_tax_summary_df.to_excel(writer, sheet_name="Spain Fees Summary", index=False)

            if afrr_method_summary_df is not None:
                afrr_method_summary_df.to_excel(writer, sheet_name="aFRR_Method_Summary", index=False)

            if afrr_block_selection_df is not None:
                afrr_block_selection_df.to_excel(writer, sheet_name="aFRR_Block_Selection", index=False)

            if afrr_afry_heuristic_audit_df is not None:
                afrr_afry_heuristic_audit_df.to_excel(writer, sheet_name="aFRR_AFRY_Audit", index=False)

            if afrr_milp_audit_df is not None:
                afrr_milp_audit_df.to_excel(writer, sheet_name="aFRR_MILP_Audit", index=False)

    # IMPORTANT:
    # Do NOT use xlsxwriter constant_memory=True with pandas.to_excel.
    # Pandas writes DataFrames column-by-column, while xlsxwriter constant-memory
    # mode requires strictly row-by-row writes. The combination silently creates
    # workbooks where most cells are blank and only the last row/column values
    # appear, which is why Monthly/Summary/Inputs looked empty after export.
    #
    # Normal xlsxwriter mode preserves all DataFrame values. openpyxl remains a
    # fallback for environments where xlsxwriter is not installed.
    output = io.BytesIO()
    try:
        _write_excel(output, engine="xlsxwriter", engine_kwargs=None)
    except Exception:
        output = io.BytesIO()
        _write_excel(output, engine="openpyxl", engine_kwargs=None)

    return output.getvalue()
