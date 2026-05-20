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

def _validate_array_length(arr: np.ndarray, name: str, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{name} doit contenir exactement {expected_len} valeurs. Reçu: {len(arr)}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non numériques ou infinies.")
    return arr
