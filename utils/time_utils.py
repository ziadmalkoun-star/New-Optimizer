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

def build_quarter_hour_index(year: int = DEFAULT_YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00:00", periods=QH_PER_YEAR, freq="15min")
