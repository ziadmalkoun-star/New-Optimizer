from pathlib import Path

HOURS_PER_YEAR = 8760
QH_PER_HOUR = 4
QH_PER_YEAR = 35040
QH_DT_HOURS = 0.25
DEFAULT_YEAR = 2025
PV_ZERO_TOLERANCE_MWH = 1e-6

# All market and aFRR price datasets must be uploaded by the user in the app.
APP_DIR = Path(__file__).resolve().parent
BUILTIN_CURTAILMENT_CURVE = APP_DIR / "Curtailment_Curve.xlsx"

def _open_builtin_file(path: Path, label: str):
    """Open the optional external curtailment file placed next to this script."""
    if not path.exists():
        raise FileNotFoundError(f"Required file '{path.name}' for {label} was not found next to the script. Please upload it instead.")
    return path.open("rb")
