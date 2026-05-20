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

def _read_single_column_csv(uploaded_file, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier CSV fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    filename = str(getattr(uploaded_file, "name", "")).lower()
    if filename.endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        df = pd.read_excel(uploaded_file, header=None)
        numeric_cols = []
        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) >= expected_len:
                numeric_cols.append(values)
        if not numeric_cols:
            raise ValueError(f"Le fichier Excel doit contenir exactement {expected_len} valeurs numériques dans une colonne. Reçu: aucune colonne exploitable.")
        arr = np.asarray(numeric_cols[-1][:expected_len], dtype=float)
        if len(arr) != expected_len:
            raise ValueError(f"Le fichier Excel doit contenir exactement {expected_len} valeurs numériques. Reçu: {len(arr)}.")
        if np.any(~np.isfinite(arr)):
            raise ValueError("Le fichier Excel contient des valeurs non finies.")
        return arr

    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = str(raw)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Le CSV est vide.")

    values = []
    bad_rows = []

    for i, line in enumerate(lines):
        cleaned = line.strip().strip('"').strip("'").replace(",", ".")
        try:
            values.append(float(cleaned))
        except ValueError:
            bad_rows.append(i)

    if len(bad_rows) == 1 and bad_rows[0] == 0 and len(values) == expected_len:
        return np.asarray(values, dtype=float)

    if bad_rows:
        raise ValueError(
            f"Le CSV contient des valeurs non numériques dans la première colonne. "
            f"Lignes problématiques: {bad_rows[:10]}"
        )

    if len(values) != expected_len:
        raise ValueError(
            f"Le CSV doit contenir exactement {expected_len} lignes numériques. "
            f"Reçu: {len(values)}."
        )

    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Le CSV contient des valeurs non finies.")
    return arr

def _read_single_column_csv_qh(uploaded_file, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    return _read_single_column_csv(uploaded_file, expected_len=expected_len)

def read_monthly_curtailment_excel(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier Excel de courbe de curtailment fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_excel(uploaded_file, header=None)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

    if len(values) != 12:
        raise ValueError(f"La courbe de curtailment mensuelle doit contenir exactement 12 valeurs. Reçu: {len(values)}.")

    return values

def read_bess_degradation_excel(uploaded_file, project_lifetime_years: int, initial_bess_mwh: float) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if uploaded_file is None:
        degradation_pct = np.full(project_lifetime_years, 100.0, dtype=float)
    else:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        df = pd.read_excel(uploaded_file, header=None)
        degradation_pct = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

        if len(degradation_pct) < project_lifetime_years:
            raise ValueError(
                f"La courbe de dégradation BESS doit contenir au moins {project_lifetime_years} valeurs. "
                f"Reçu: {len(degradation_pct)}."
            )

        degradation_pct = degradation_pct[:project_lifetime_years]

    if len(degradation_pct) == 0:
        raise ValueError("La courbe de dégradation BESS est vide.")

    if degradation_pct[0] <= 1.5:
        degradation_pct = degradation_pct * 100.0

    degraded_mwh = np.zeros(project_lifetime_years, dtype=float)
    degraded_mwh[0] = float(initial_bess_mwh) * degradation_pct[0] / 100.0
    
    for y in range(1, project_lifetime_years):
        degraded_mwh[y] = degraded_mwh[y - 1] * degradation_pct[y] / 100.0

    degradation_df = pd.DataFrame({
        "Year": np.arange(1, project_lifetime_years + 1),
        "Degradation_pct": degradation_pct,
        "BESS_energy_mwh": degraded_mwh,
    })

    return degradation_pct, degraded_mwh, degradation_df
