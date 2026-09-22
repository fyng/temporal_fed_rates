"""Build, save and load the analysis frame consumed by all plotting and models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from . import fred, registry, rules, sources, transforms

__all__ = [
    "COLUMNS",
    "build_frame",
    "frame_path",
    "load_frame",
    "recession_spans",
    "save_frame",
    "tidy",
]

Vintage = fred.Vintage

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

_TARGET_SPLICE = "2008-12-16"  # DFEDTAR ends 2008-12-15; DFEDTARU/L begin 2008-12-16
_IOR_SPLICE = "2021-07-29"  # IOER ends 2021-07-28; IORB begins 2021-07-29

# --- Column contract ---------------------------------------------------------

_RULE_BASE: dict[str, str] = {
    "taylor93": "Taylor (1993)",
    "balanced": "Balanced approach",
    "shortfalls": "Balanced approach, shortfalls",
    "inertial": "Inertial rule",
    "first_diff": "First-difference rule",
}

COLUMNS: dict[str, str] = {
    # raw registry columns
    "ffr": "Effective fed funds rate, monthly mean",
    "ffr_m": "Effective fed funds rate (FRED monthly series)",
    "effr": "EFFR, monthly mean from 2000",
    "target_old": "Fed funds target, single point, to 2008-12-15",
    "target_upper": "Target range upper bound, end of month",
    "target_lower": "Target range lower bound, end of month",
    "ioer": "Interest on excess reserves, end of month",
    "iorb": "Interest on reserve balances, end of month",
    "onrrp": "Overnight reverse repo award rate, end of month",
    "cpi": "CPI, all items, index",
    "cpi_core": "Core CPI, index",
    "pce": "PCE price index, index",
    "pce_core": "Core PCE price index, index",
    "unrate": "Unemployment rate",
    "nrou": "Natural rate of unemployment, CBO",
    "gdp": "Real GDP, $bn 2017 chained",
    "gdp_pot": "Real potential GDP, CBO",
    "payems": "Nonfarm payrolls, thousands",
    "claims": "Initial jobless claims, monthly mean",
    "bei10": "10-year breakeven inflation, monthly mean",
    "bei5": "5-year breakeven inflation, monthly mean",
    "fwd5y5y": "5-year 5-year forward inflation, monthly mean",
    "mich": "Michigan inflation expectations, end of month",
    "spread_10y2y": "10y-2y Treasury spread (FRED), monthly mean",
    "dgs10": "10-year Treasury yield, monthly mean",
    "dgs2": "2-year Treasury yield, monthly mean",
    "nfci": "Chicago Fed financial conditions, end of month",
    "anfci": "Chicago Fed adjusted financial conditions, end of month",
    "vix": "VIX, monthly mean",
    "recession": "NBER recession indicator, 0/1",
    # spliced
    "target_mid": "Target midpoint across the 2008 splice",
    "ior": "Rate paid on reserves across the 2021 splice, end of month",
    "shadow_rate": "Wu-Xia shadow rate, 1990-01 to 2022-02",
    "policy_rate": "Effective rate with Wu-Xia substituted at the ELB",
    # derived
    "pce_core_yoy": "Core PCE inflation, y/y",
    "cpi_core_yoy": "Core CPI inflation, y/y",
    "pi_gap": "Core PCE inflation gap vs the 2% target",
    "u_gap": "Unemployment gap, u - u*",
    "y_gap": "Output gap, 100 log(y/y*)",
    "real_rate": "Policy rate minus core PCE inflation",
    "term_spread": "10-year minus 2-year yield",
    "r_star": "Neutral real rate used by the rules",
}
for _key, _label in _RULE_BASE.items():
    COLUMNS[f"rule_{_key}"] = f"{_label} prescription, floored at the ELB"
    COLUMNS[f"rule_{_key}_unfloored"] = f"{_label} prescription, unfloored"

# Labels and groups for columns not in the registry.
_SPLICED_LABELS: dict[str, str] = {
    "target_mid": "Fed funds target midpoint",
    "ior": "Rate paid on reserves",
    "shadow_rate": "Wu-Xia shadow rate",
    "policy_rate": "Policy rate",
}
_DERIVED_LABELS: dict[str, str] = {
    "pce_core_yoy": "Core PCE inflation, y/y",
    "cpi_core_yoy": "Core CPI inflation, y/y",
    "pi_gap": "Inflation gap vs 2%",
    "u_gap": "Unemployment gap",
    "y_gap": "Output gap",
    "real_rate": "Ex-post real rate",
    "term_spread": "Term spread, 10y-2y",
    "r_star": "Neutral rate r*",
}


def _column_meta() -> dict[str, tuple[str, str]]:
    """Map frame column to (label, group) for every emitted column.

    Returns:
        Dict of column name to label and tidy group.
    """
    out = {s.column: (s.name, s.group) for s in registry.REGISTRY.values()}
    out.update((c, (label, "policy")) for c, label in _SPLICED_LABELS.items())
    out.update((c, (label, "derived")) for c, label in _DERIVED_LABELS.items())
    for key, label in _RULE_BASE.items():
        out[f"rule_{key}"] = (label, "rule")
        out[f"rule_{key}_unfloored"] = (f"{label}, unfloored", "rule")
    return out


# --- Frame assembly ----------------------------------------------------------


def build_frame(
    freq: Literal["m", "q"] = "m",
    *,
    start: str = "1960-01-01",
    end: str | None = None,
    r_star: float | pd.Series | Literal["hlw"] = 2.0,
    shadow: bool = True,
    include_rules: bool = True,
    vintage: Vintage = None,
    force: bool = False,
) -> pd.DataFrame:
    """Assemble the analysis frame from cached FRED series and policy rules.

    Args:
        freq: "m" for month-start or "q" for quarter-start index.
        start: Inclusive start of the frame; YoY columns need a year of warmup.
        end: Inclusive end; when None, the frame stops at the last month or
            quarter with an observed effective fed funds rate.
        r_star: Neutral real rate: a constant, a series, or "hlw" for the
            Holston-Laubach-Williams estimate.
        shadow: Substitute the Wu-Xia shadow rate for the effective rate
            wherever it sits at or below the ELB.
        include_rules: Add the ``rule_*`` prescription columns.
        vintage: Passed through to ``fred.fetch_many``.
        force: Refetch even when the cache is fresh.

    Returns:
        Wide float64 frame on a DatetimeIndex named ``date``; columns are
        documented in ``COLUMNS``.

    Raises:
        ValueError: If ``r_star`` is a string other than "hlw".
    """
    raw = fred.fetch_many(registry.ids(), vintage=vintage, force=force)
    series: dict[str, pd.Series] = {}
    how: dict[str, str] = {}
    for sid, s in raw.items():
        meta = registry.get(sid)
        series[meta.column] = s
        how[meta.column] = meta.agg
    frame = transforms.align(series, freq=freq, how=how, start=start, end=end)
    if end is None:  # CBO projections run ~10y ahead; stop at observed policy
        frame = frame.loc[: frame["ffr"].last_valid_index()]
    frame = _add_spliced(frame, shadow=shadow)
    frame = _add_derived(frame)
    r_star_used = _resolve_r_star(r_star, freq, frame.index)
    frame["r_star"] = r_star_used
    if include_rules:
        frame = frame.join(rules.prescriptions(frame, r_star=r_star_used))
    return frame.astype("float64")


def _add_spliced(frame: pd.DataFrame, *, shadow: bool) -> pd.DataFrame:
    """Add target_mid, ior, shadow_rate and policy_rate columns.

    Args:
        frame: Aligned raw frame.
        shadow: Substitute the shadow rate at the ELB.

    Returns:
        The frame with the spliced columns appended.
    """
    mid = (frame["target_upper"] + frame["target_lower"]) / 2.0
    frame["target_mid"] = transforms.splice(mid, frame["target_old"], at=_TARGET_SPLICE)
    frame["ior"] = transforms.splice(frame["iorb"], frame["ioer"], at=_IOR_SPLICE)
    frame["shadow_rate"] = sources.wu_xia().reindex(frame.index)
    if shadow:
        mask = frame["ffr"].le(rules.ELB) & frame["shadow_rate"].notna()
        frame["policy_rate"] = transforms.splice(
            frame["ffr"], frame["shadow_rate"], where=mask
        )
    else:
        frame["policy_rate"] = frame["ffr"].rename("policy_rate")
    return frame


def _add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add inflation, gap and spread columns.

    Args:
        frame: Frame with the spliced columns in place.

    Returns:
        The frame with the derived columns appended.
    """
    frame["pce_core_yoy"] = transforms.yoy(frame["pce_core"])
    frame["cpi_core_yoy"] = transforms.yoy(frame["cpi_core"])
    frame["pi_gap"] = transforms.inflation_gap(frame["pce_core"])
    frame["u_gap"] = transforms.unemployment_gap(frame["unrate"], frame["nrou"])
    frame["y_gap"] = transforms.output_gap(frame["gdp"], frame["gdp_pot"])
    frame["real_rate"] = transforms.real_rate(frame["policy_rate"], frame["pce_core_yoy"])
    frame["term_spread"] = transforms.term_spread(frame["dgs10"], frame["dgs2"])
    return frame


def _resolve_r_star(
    r_star: float | pd.Series | Literal["hlw"],
    freq: Literal["m", "q"],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Broadcast the chosen r* onto the frame index.

    Args:
        r_star: Constant, series, or "hlw".
        freq: Frame frequency.
        index: Frame index to land on.

    Returns:
        Series of neutral real rates on ``index``.

    Raises:
        ValueError: If ``r_star`` is a string other than "hlw".
    """
    if isinstance(r_star, str):
        if r_star != "hlw":
            raise ValueError(f'r_star must be a float, a Series or "hlw", got {r_star!r}')
        s = sources.hlw()
        return (
            transforms.quarterly_to_monthly(s, "ffill").reindex(index)
            if freq == "m"
            else s.reindex(index)
        ).astype("float64")
    if isinstance(r_star, pd.Series):
        return r_star.reindex(index).astype("float64")
    return pd.Series(float(r_star), index=index)


# --- Persistence -------------------------------------------------------------


def frame_path(name: str = "analysis_monthly") -> Path:
    """Resolve the processed-CSV path for a named frame.

    Args:
        name: Frame name.

    Returns:
        Path to ``data/processed/<name>.csv``.
    """
    return PROCESSED_DIR / f"{name}.csv"


def save_frame(frame: pd.DataFrame, name: str = "analysis_monthly") -> Path:
    """Write a frame to the processed directory as CSV.

    Args:
        frame: Frame with a DatetimeIndex.
        name: Frame name.

    Returns:
        Path to the written file.
    """
    path = frame_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.index.name = "date"
    out.to_csv(path)
    return path


def load_frame(name: str = "analysis_monthly", *, rebuild: bool = False, **kw) -> pd.DataFrame:
    """Load a named frame, building it first when absent or ``rebuild`` is set.

    Args:
        name: Frame name.
        rebuild: Rebuild and overwrite even when the CSV exists.
        **kw: Forwarded to ``build_frame``; only used when the frame is built.

    Returns:
        The frame as float64 on a DatetimeIndex named ``date``.
    """
    path = frame_path(name)
    if rebuild or not path.exists():
        save_frame(build_frame(**kw), name)
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame.astype("float64")


# --- Tidy and furniture ------------------------------------------------------


def tidy(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Melt a frame to a long form carrying registry labels and groups.

    Args:
        frame: Frame produced by ``build_frame``.
        columns: Columns to keep; all documented columns when None.

    Returns:
        Long frame with exactly ``date, series, value, label, group``.

    Raises:
        KeyError: If a requested column has no label or group.
    """
    cols = list(frame.columns) if columns is None else list(columns)
    meta = _column_meta()
    unknown = [c for c in cols if c not in meta]
    if unknown:
        raise KeyError(f"columns without a label or group: {unknown}")
    long = (
        frame[cols]
        .reset_index(names="date")
        .melt(id_vars="date", var_name="series", value_name="value")
    )
    long["label"] = long["series"].map({c: m[0] for c, m in meta.items()})
    long["group"] = long["series"].map({c: m[1] for c, m in meta.items()})
    return long[["date", "series", "value", "label", "group"]]


def recession_spans(
    frame: pd.DataFrame | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous NBER recession runs as (start, end) pairs.

    Args:
        frame: Frame with a ``recession`` column; ``load_frame()`` when None.

    Returns:
        Ordered, non-overlapping (start, end) timestamp pairs for each run of
        1s, shaped for ``theme.period_shading``.
    """
    if frame is None:
        frame = load_frame()
    flag = frame["recession"].eq(1)
    runs = flag.ne(flag.shift()).cumsum()
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, block in frame.loc[flag].groupby(runs.loc[flag]):
        idx = block.index
        spans.append((idx[0], idx[-1]))
    return spans
