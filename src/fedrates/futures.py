"""Fed funds futures benchmark: market expectations, not a fitted model.

The SF Fed monetary-policy-surprises workbook (Bauer and Swanson 2023)
reports 30-minute changes in the current- and next-month fed funds futures
rates, FF1 and FF2, around each FOMC decision. :func:`kuttner` turns them
into Kuttner (2001) policy surprises in basis points per the
Gurkaynak-Sack-Swanson convention, :func:`expected_proba` maps the
market-expected change onto the five decision classes FedWatch-style, and
:func:`benchmark` assembles the record-shaped output to score against the
decision model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import fomc

__all__ = [
    "CLASSES",
    "SURPRISES_URL",
    "benchmark",
    "expected_proba",
    "kuttner",
    "surprises",
]

CLASSES: tuple[str, ...] = fomc.CLASSES

SURPRISES_URL = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"
_SHEET = "FOMC (update 2023)"
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_CACHE = _RAW_DIR / "monetary-policy-surprises-data.xlsx"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research data collection"}


def _download(path: Path, *, force: bool, max_age_days: float) -> None:
    """Fetch the workbook into ``path`` unless a fresh copy is cached.

    Args:
        path: Cache destination.
        force: Re-download even when the cached file is fresh.
        max_age_days: Cache age above which the file is re-downloaded.

    Raises:
        FileNotFoundError: If the download fails and no cached file exists.
    """
    fresh = path.exists() and (time.time() - path.stat().st_mtime) <= max_age_days * 86400
    if fresh and not force:
        return
    try:
        response = requests.get(SURPRISES_URL, headers=_HEADERS, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            return
        raise FileNotFoundError(
            f"Could not download {SURPRISES_URL}; place the file manually at {path}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(path)


def surprises(path: Path | None = None, *, force: bool = False,
              max_age_days: float = 30.0) -> pd.DataFrame:
    """FF1/FF2 futures changes around each FOMC decision, from the SF Fed.

    Args:
        path: Local xlsx override; ``data/raw/monetary-policy-surprises-data.xlsx``
            when None.
        force: Re-download even when the cached file is fresh.
        max_age_days: Cache age above which the file is re-downloaded.

    Returns:
        Frame indexed by decision date with the ``ff1`` and ``ff2`` columns,
        30-minute changes in percentage points; duplicated dates keep the
        last row.
    """
    path = _CACHE if path is None else Path(path)
    _download(path, force=force, max_age_days=max_age_days)
    raw = pd.read_excel(path, sheet_name=_SHEET)
    out = pd.DataFrame(
        {"ff1": pd.to_numeric(raw["FF1"]).to_numpy(),
         "ff2": pd.to_numeric(raw["FF2"]).to_numpy()},
        index=pd.DatetimeIndex(pd.to_datetime(raw["Date"]), name="date"),
    )
    return out[~out.index.duplicated(keep="last")]


def kuttner(dates, ff1, ff2) -> pd.Series:
    """Kuttner (2001) policy surprise in basis points.

    Gurkaynak-Sack-Swanson convention: a decision on day d of a D-day month
    moves the month-end-settled current futures contract by d's share of the
    move, so the surprise is ``100 * FF1 * D / (D - d)``. Meetings in the
    last 7 days of the month read the next-month contract instead, unscaled
    (``100 * FF2``).

    Args:
        dates: Decision dates.
        ff1: FF1 30-minute changes in percentage points, aligned with ``dates``.
        ff2: FF2 30-minute changes in percentage points, aligned with ``dates``.

    Returns:
        Surprise in basis points on ``dates``; NaN where the contract used is
        missing.
    """
    dates = pd.DatetimeIndex(dates)
    day = dates.day.to_numpy()
    n_days = dates.days_in_month.to_numpy()
    end_of_month = (n_days - day) < 7
    denom = np.where(end_of_month, 1, n_days - day)
    ff1 = np.asarray(ff1, dtype=float)
    ff2 = np.asarray(ff2, dtype=float)
    bp = np.where(end_of_month, 100.0 * ff2, 100.0 * ff1 * n_days / denom)
    return pd.Series(bp, index=dates, name="surprise_bp")


def expected_proba(expected_bp: pd.Series | np.ndarray) -> pd.DataFrame:
    """FedWatch-style class probabilities from expected changes, in basis points.

    The expectation is clipped to [-100, +100] and split linearly between the
    two adjacent multiples of 25bp that bracket it; a multiple of 25 maps to
    a point mass, with -50 or below read as cut50+, -25 as cut25, 0 as hold,
    +25 as hike25 and +50 or above as hike50+.

    Args:
        expected_bp: Market-expected change in basis points (actual minus
            surprise).

    Returns:
        Frame over :data:`CLASSES` sharing the input's index; each finite row
        sums to 1, non-finite expectations give all-NaN rows.
    """
    e = np.asarray(expected_bp, dtype=float)
    index = expected_bp.index if isinstance(expected_bp, pd.Series) else None
    out = np.full((e.size, len(CLASSES)), np.nan)
    ok = np.isfinite(e)
    x = np.round(np.clip(e[ok], -100.0, 100.0), 9)
    low = np.floor(x / 25.0)
    frac = x / 25.0 - low
    rows = np.arange(x.size)
    code = np.clip(low.astype(int) + 2, 0, 4)  # class of the lower bracketing multiple
    high = np.clip(low.astype(int) + 3, 0, 4)  # class of the upper bracketing multiple
    block = np.zeros((x.size, len(CLASSES)))
    block[rows, high] = frac
    block[rows, code] += 1.0 - frac
    out[ok] = block
    return pd.DataFrame(out, index=index, columns=list(CLASSES))


def benchmark(meetings: pd.DataFrame | None = None, path: Path | None = None, *,
              force: bool = False, max_age_days: float = 30.0) -> pd.DataFrame:
    """Futures-implied class probabilities, shaped like a walk-forward output.

    The market-expected change per meeting is the realised ``size_bp`` minus
    the Kuttner surprise. Meetings the workbook lacks, or lacks a usable
    FF1/FF2 print for, are dropped.

    Args:
        meetings: Meetings frame; ``fomc.load_meetings()`` when None.
        path: Workbook path override, forwarded to :func:`surprises`.
        force: Re-download even when cached.
        max_age_days: Workbook cache age limit.

    Returns:
        Frame indexed by meeting date with the five class-probability columns
        and the realised ``label``.
    """
    meetings = fomc.load_meetings() if meetings is None else meetings
    m = meetings.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(m["date"], name="date")
    sur = surprises(path, force=force, max_age_days=max_age_days).reindex(dates)
    surprise = kuttner(dates, sur["ff1"].to_numpy(), sur["ff2"].to_numpy())
    expected = pd.Series(
        m["size_bp"].to_numpy(dtype=float) - surprise.to_numpy(), index=dates, name="expected_bp"
    )
    out = expected_proba(expected)
    out = out[out.notna().all(axis=1)]
    label = fomc.label_actions(m)
    label.index = dates
    out["label"] = label.reindex(out.index)
    return out
