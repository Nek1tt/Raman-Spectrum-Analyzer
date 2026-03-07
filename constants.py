"""
constants.py — Shared constants for Raman v10 pipeline.
"""

import re
from typing import Dict, List

# ── Class colours (shared with inference_utils) ──────────────────────────────
COLORS: Dict[str, str] = {
    "control": "tab:blue",
    "endo":    "tab:orange",
    "exo":     "tab:green",
}

# ── Dataset folder structure ──────────────────────────────────────────────────
CLASS_DIRS: Dict[str, List[str]] = {
    "control": ["mk1",   "mk2a",   "mk2b",   "mk3"],
    "endo":    ["mend1", "mend2a", "mend2b", "mend3"],
    "exo":     ["mexo1", "mexo2a", "mexo2b", "mexo3"],
}

ANIMAL_RE     = re.compile(r"(\d+[ab]?$)")
BRAIN_REGIONS = ["cerebellum", "striatum", "cortex"]
CENTER_RE     = re.compile(r"center(\d+)")
PLACE_RE      = re.compile(r"place(\d+)")

# Wavenumber ranges per spectral centre
BAND_RANGES: Dict[int, tuple] = {
    1500: (900,  2050),
    2900: (2650, 3300),
}

KNOWN_REGIONS = ["cortex", "striatum", "cerebellum", "unknown"]
