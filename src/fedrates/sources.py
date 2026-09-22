"""Loaders for the two key series not on FRED: Wu-Xia shadow rate, HLW r-star."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

__all__ = ["WU_XIA_URL", "HLW_URL", "RAW_DIR", "download", "wu_xia", "hlw"]

# --- Sources and cache -------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

WU_XIA_URL = (
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/datafiles/"
    "cqer/research/wu-xia-shadow-federal-funds-rate/WuXiaShadowRate.xlsx"
)
HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/williams/"
    "data/Holston_Laubach_Williams_current_estimates.xlsx"
)
_WU_XIA_CACHE = RAW_DIR / "WuXiaShadowRate.xlsx"
_HLW_CACHE = RAW_DIR / "Holston_Laubach_Williams_current_estimates.xlsx"

_HLW_SHEET = "HLW Estimates"
_HLW_MEASURE_COL = {"g": 2, "z": 6, "rstar": 10}
_HLW_SERIES_NAME = {"rstar": "r_star", "g": "g", "z": "z"}

# --- Download ----------------------------------------------------------------


def _is_fresh(path: Path, max_age_days: float) -> bool:
    """Report whether the cached file is younger than the limit.

    Args:
        path: Cached file.
        max_age_days: Maximum age in days.

    Returns:
        True when the file's mtime is within the limit.
    """
    return (time.time() - path.stat().st_mtime) <= max_age_days * 86400


def download(url: str, path: Path, *, force: bool = False, max_age_days: float = 30.0) -> Path:
    """Fetch ``url`` into ``path`` unless a fresh copy is already cached.

    Falls back to the cached file when a refresh fails.

    Args:
        url: Source URL.
        path: Cache destination.
        force: Re-download even when the cached file is fresh.
        max_age_days: Cache age above which the file is re-downloaded.

    Returns:
        Path to the cached file.

    Raises:
        FileNotFoundError: If the download fails and no cached file exists.
    """
    path = Path(path)
    if path.exists() and not force and _is_fresh(path, max_age_days):
        return path
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Could not download {url}; place the file manually at {path}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


# --- Wu-Xia ------------------------------------------------------------------


def wu_xia(
    *, path: Path | None = None, force: bool = False, max_age_days: float = 30.0
) -> pd.Series:
    """Wu-Xia shadow federal funds rate, monthly, from the Atlanta Fed CenFIS.

    Not on FRED. The authors no longer update the series (ends 2022-02); splice
    with the effective federal funds rate outside ELB windows. Values before
    1990-01 are NaN: the model sample starts there.

    Args:
        path: Local xlsx override for when the URL moves; defaults to
            ``RAW_DIR / "WuXiaShadowRate.xlsx"``.
        force: Re-download even when the cached file is fresh.
        max_age_days: Cache age above which the file is re-downloaded.

    Returns:
        Monthly shadow rate in percent; DatetimeIndex ``date`` at month starts.

    Raises:
        FileNotFoundError: If the file is absent and cannot be downloaded.
    """
    path = _WU_XIA_CACHE if path is None else Path(path)
    download(WU_XIA_URL, path, force=force, max_age_days=max_age_days)
    return _wu_xia_series(path)


def _wu_xia_series(path: Path) -> pd.Series:
    """Parse the Wu-Xia xlsx into a monthly series.

    Args:
        path: Local WuXiaShadowRate.xlsx.

    Returns:
        Monthly shadow rate; DatetimeIndex ``date`` at month starts.
    """
    # Observed 2026-09: sheet "Data"; header row 1; col A month dates, B FFR, C shadow rate.
    raw = pd.read_excel(path, sheet_name="Data", header=None, skiprows=1, usecols=[0, 2])
    dates = pd.to_datetime(raw[0], errors="coerce").dt.to_period("M").dt.to_timestamp()
    values = pd.to_numeric(raw[2], errors="coerce").astype("float64")
    keep = dates.notna()
    index = pd.DatetimeIndex(dates[keep].to_numpy(), name="date")
    return pd.Series(values[keep].to_numpy(dtype="float64"), index=index, name="shadow_rate")


# --- HLW ---------------------------------------------------------------------


def hlw(
    *,
    path: Path | None = None,
    measure: Literal["rstar", "g", "z"] = "rstar",
    force: bool = False,
    max_age_days: float = 30.0,
) -> pd.Series:
    """Holston-Laubach-Williams natural rate of interest, quarterly, from the NY Fed.

    Not on FRED. US panel of the current-estimates workbook.

    Args:
        path: Local xlsx override for when the URL moves; defaults to
            ``RAW_DIR / "Holston_Laubach_Williams_current_estimates.xlsx"``.
        measure: "rstar", "g" (trend growth) or "z" (residual demand factor).
        force: Re-download even when the cached file is fresh.
        max_age_days: Cache age above which the file is re-downloaded.

    Returns:
        Quarterly US measure; DatetimeIndex ``date`` at quarter starts.

    Raises:
        KeyError: If ``measure`` is not one of "rstar", "g", "z".
        FileNotFoundError: If the file is absent and cannot be downloaded.
    """
    if measure not in _HLW_MEASURE_COL:
        raise KeyError(f"measure must be one of {sorted(_HLW_MEASURE_COL)}, got {measure!r}")
    path = _HLW_CACHE if path is None else Path(path)
    download(HLW_URL, path, force=force, max_age_days=max_age_days)
    return _hlw_series(path, measure)


def _hlw_series(path: Path, measure: str) -> pd.Series:
    """Parse the HLW xlsx into a quarterly series for one US measure.

    Args:
        path: Local Holston_Laubach_Williams_current_estimates.xlsx.
        measure: Key of ``_HLW_MEASURE_COL``.

    Returns:
        Quarterly series; DatetimeIndex ``date`` at quarter starts.
    """
    # Observed 2026-09: sheet "HLW Estimates"; rows 1-6 titles/heads; col A dates,
    # US g/z/rstar at C/G/K, "NA" strings for gaps.
    raw = pd.read_excel(
        path, sheet_name=_HLW_SHEET, header=None, skiprows=6, usecols=[0, _HLW_MEASURE_COL[measure]]
    )
    dates = pd.to_datetime(raw[0], errors="coerce").dt.to_period("Q").dt.to_timestamp()
    values = pd.to_numeric(raw[_HLW_MEASURE_COL[measure]], errors="coerce")
    keep = dates.notna() & values.notna()
    index = pd.DatetimeIndex(dates[keep].to_numpy(), name="date")
    return pd.Series(
        values[keep].to_numpy(dtype="float64"), index=index, name=_HLW_SERIES_NAME[measure]
    )
