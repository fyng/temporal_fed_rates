"""Monetary policy rules: prescribed policy rates from an analysis frame.

Inputs follow ``transforms`` sign conventions: ``u_gap = u − u*`` (positive
means slack) and ``y_gap = 100·log(y/y*)`` (positive means above potential),
so gap terms enter as ``−a·u_gap`` and ``+a·y_gap``. π is twelve-month core
PCE inflation in percent and r* the neutral real rate.

Defaults for ``a_gap=None``: taylor1993 uses 1.0 on the unemployment gap and
0.5 on the output gap; balanced_approach uses 2.0 and 1.0.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import pandas as pd

from fedrates.transforms import _ppy

__all__ = [
    "RULES",
    "ELB",
    "taylor1993",
    "balanced_approach",
    "balanced_approach_shortfalls",
    "inertial",
    "first_difference",
    "prescriptions",
]

RStar = float | pd.Series
Gap = Literal["output", "unemployment"]

# Midpoint of the 0-0.25% fed funds target range.
ELB = 0.125

_DEFAULT_COLS = {
    "pi": "pce_core_yoy",
    "u_gap": "u_gap",
    "y_gap": "y_gap",
    "rate": "policy_rate",
}
_GAP: dict[str, tuple[str, float]] = {
    "unemployment": ("u_gap", -1.0),
    "output": ("y_gap", 1.0),
}
_A_GAP_DEFAULT = {
    ("taylor1993", "unemployment"): 1.0,
    ("taylor1993", "output"): 0.5,
    ("balanced_approach", "unemployment"): 2.0,
    ("balanced_approach", "output"): 1.0,
}


# --- Frame helpers -----------------------------------------------------------


def _names(cols: Mapping[str, str] | None) -> dict[str, str]:
    """Merge a column remap over the defaults.

    Args:
        cols: User remap; None keeps the defaults.

    Returns:
        Canonical rule input to frame column name.
    """
    return {**_DEFAULT_COLS, **(cols or {})}


def _get(frame: pd.DataFrame, name: str, key: str) -> pd.Series:
    """Fetch a rule input column by its resolved name.

    Args:
        frame: Analysis frame.
        name: Resolved frame column name.
        key: Canonical rule input the column supplies.

    Returns:
        The column.

    Raises:
        KeyError: If the frame lacks the column.
    """
    if name not in frame.columns:
        raise KeyError(f"frame lacks column {name!r} (rule input {key!r})")
    return frame[name]


def _broadcast(r_star: RStar, index: pd.DatetimeIndex) -> pd.Series:
    """Broadcast r* onto a frame index.

    Args:
        r_star: Constant or series of neutral real rates in percent.
        index: Frame index to broadcast onto.

    Returns:
        Series on ``index``; a shorter series yields NaN where absent.
    """
    if isinstance(r_star, pd.Series):
        return r_star.reindex(index).astype(float)
    return pd.Series(float(r_star), index=index)


def _gap_input(frame: pd.DataFrame, names: Mapping[str, str], gap: Gap) -> pd.Series:
    """Signed activity-gap input: −u_gap or +y_gap.

    Args:
        frame: Analysis frame.
        names: Resolved column mapping.
        gap: Which gap to use.

    Returns:
        The signed gap series.

    Raises:
        ValueError: If ``gap`` is not "unemployment" or "output".
    """
    if gap not in _GAP:
        raise ValueError(f"gap must be one of {sorted(_GAP)}, got {gap!r}")
    key, sign = _GAP[gap]
    return sign * _get(frame, names[key], key)


# --- Linear rules ------------------------------------------------------------


def _linear_rule(
    frame: pd.DataFrame,
    r_star: RStar,
    *,
    target: float,
    a_pi: float,
    a_gap: float | None,
    gap: Gap,
    cols: Mapping[str, str] | None,
    name: str,
) -> pd.Series:
    """r* + π + a_pi·(π − π*) + a·gap, with a from the defaults when None.

    Args:
        frame: Analysis frame.
        r_star: Neutral real rate in percent, constant or series.
        target: Inflation target in percent.
        a_pi: Inflation-gap coefficient.
        a_gap: Activity-gap coefficient; None uses the rule-and-gap default.
        gap: Activity gap to use.
        cols: Column remap over the defaults.
        name: Rule name keying the defaults table.

    Returns:
        Prescribed policy rate on the frame index.
    """
    names = _names(cols)
    gap_term = _gap_input(frame, names, gap)
    a = a_gap if a_gap is not None else _A_GAP_DEFAULT[(name, gap)]
    pi = _get(frame, names["pi"], "pi")
    r = _broadcast(r_star, frame.index)
    return r + pi + a_pi * (pi - target) + a * gap_term


def taylor1993(
    frame: pd.DataFrame,
    r_star: RStar = 2.0,
    *,
    target: float = 2.0,
    a_pi: float = 0.5,
    a_gap: float | None = None,
    gap: Gap = "unemployment",
    cols: Mapping[str, str] | None = None,
) -> pd.Series:
    """Taylor (1993) prescription.

    Args:
        frame: Analysis frame with inflation and an activity gap.
        r_star: Neutral real rate in percent, constant or series.
        target: Inflation target in percent.
        a_pi: Inflation-gap coefficient.
        a_gap: Activity-gap coefficient; None uses the rule-and-gap default.
        gap: Activity gap to use.
        cols: Column remap over the defaults.

    Returns:
        Prescribed policy rate on the frame index.
    """
    return _linear_rule(
        frame, r_star, target=target, a_pi=a_pi, a_gap=a_gap, gap=gap, cols=cols,
        name="taylor1993",
    )


def balanced_approach(
    frame: pd.DataFrame,
    r_star: RStar = 2.0,
    *,
    target: float = 2.0,
    a_pi: float = 0.5,
    a_gap: float | None = None,
    gap: Gap = "unemployment",
    cols: Mapping[str, str] | None = None,
) -> pd.Series:
    """Balanced-approach prescription: equal weight on both gaps.

    Args:
        frame: Analysis frame with inflation and an activity gap.
        r_star: Neutral real rate in percent, constant or series.
        target: Inflation target in percent.
        a_pi: Inflation-gap coefficient.
        a_gap: Activity-gap coefficient; None uses the rule-and-gap default.
        gap: Activity gap to use.
        cols: Column remap over the defaults.

    Returns:
        Prescribed policy rate on the frame index.
    """
    return _linear_rule(
        frame, r_star, target=target, a_pi=a_pi, a_gap=a_gap, gap=gap, cols=cols,
        name="balanced_approach",
    )


def balanced_approach_shortfalls(
    frame: pd.DataFrame,
    r_star: RStar = 2.0,
    *,
    target: float = 2.0,
    a_pi: float = 0.5,
    a_gap: float = 2.0,
    cols: Mapping[str, str] | None = None,
) -> pd.Series:
    """Balanced approach reacting only to labour shortfalls.

    The gap term is a_gap·min(−u_gap, 0): slack still eases, a hot labour
    market adds nothing. The inflation term is unchanged.

    Args:
        frame: Analysis frame with inflation and u_gap.
        r_star: Neutral real rate in percent, constant or series.
        target: Inflation target in percent.
        a_pi: Inflation-gap coefficient.
        a_gap: Coefficient on the shortfall term min(−u_gap, 0).
        cols: Column remap over the defaults.

    Returns:
        Prescribed policy rate on the frame index.
    """
    names = _names(cols)
    pi = _get(frame, names["pi"], "pi")
    u_gap = _get(frame, names["u_gap"], "u_gap")
    shortfall = np.minimum(-u_gap, 0.0)
    return _broadcast(r_star, frame.index) + pi + a_pi * (pi - target) + a_gap * shortfall


# --- Lagged rules ------------------------------------------------------------


def inertial(
    frame: pd.DataFrame,
    r_star: RStar = 2.0,
    *,
    rho: float = 0.85,
    base: Callable[..., pd.Series] = balanced_approach,
    mode: Literal["actual", "recursive"] = "actual",
    rate_col: str = "policy_rate",
    cols: Mapping[str, str] | None = None,
    **base_kwargs: object,
) -> pd.Series:
    """Smooth a base rule: rho·R_{t−1} + (1 − rho)·base(...).

    Under ``mode="actual"`` R_{t−1} is the observed lagged policy rate, as
    published in the Fed's Monetary Policy Report. Under ``mode="recursive"``
    the rule's own prescription is iterated forward from a seed of the first
    observed rate; periods where the base prescription is NaN hold the prior
    value.

    Args:
        frame: Analysis frame.
        r_star: Neutral real rate in percent, constant or series.
        rho: Smoothing parameter on the lagged rate.
        base: Rule supplying the unconstrained prescription; must accept
            ``frame``, ``r_star`` and ``cols``.
        mode: "actual" conditions on the observed rate; "recursive" iterates
            the rule's own path.
        rate_col: Observed policy-rate column; ``cols["rate"]`` overrides.
        cols: Column remap, passed through to ``base``.
        **base_kwargs: Forwarded to ``base``.

    Returns:
        Prescribed policy rate on the frame index; NaN in the first period
        under "actual".

    Raises:
        ValueError: If ``mode`` is not "actual" or "recursive".
    """
    if mode not in ("actual", "recursive"):
        raise ValueError(f"mode must be 'actual' or 'recursive', got {mode!r}")
    rate = _get(frame, (cols or {}).get("rate", rate_col), "rate")
    base_val = base(frame, r_star=r_star, cols=cols, **base_kwargs)
    if mode == "actual":
        return rho * rate.shift(1) + (1.0 - rho) * base_val
    seed = rate.dropna()
    prev = float(seed.iloc[0]) if len(seed) else float("nan")
    b = base_val.to_numpy(dtype=float)
    out = np.empty(b.size, dtype=float)
    for t in range(b.size):
        if t > 0 and np.isfinite(b[t]):
            prev = rho * prev + (1.0 - rho) * b[t]
        out[t] = prev
    return pd.Series(out, index=frame.index)


def first_difference(
    frame: pd.DataFrame,
    *,
    target: float = 2.0,
    a_pi: float = 0.5,
    a_gap: float = 1.0,
    lag: int | None = None,
    rate_col: str = "policy_rate",
    gap: Gap = "unemployment",
    cols: Mapping[str, str] | None = None,
) -> pd.Series:
    """First-difference rule: R_{t−1} + a_pi·(π − π*) + a_gap·Δ_lag(gap).

    The Fed's published version uses a_pi=0.5, a_gap=1.0 on the four-quarter
    change; the older damped variant is a_pi=a_gap=0.1.

    Args:
        frame: Analysis frame.
        target: Inflation target in percent.
        a_pi: Inflation-gap coefficient.
        a_gap: Gap-change coefficient.
        lag: Change horizon in periods; inferred from the index (12 monthly,
            4 quarterly) when None.
        rate_col: Observed policy-rate column; ``cols["rate"]`` overrides.
        gap: Activity gap to use.
        cols: Column remap over the defaults.

    Returns:
        Prescribed policy rate on the frame index; NaN until the first
        observed rate and the first full change horizon.
    """
    names = _names(cols)
    rate = _get(frame, (cols or {}).get("rate", rate_col), "rate")
    pi = _get(frame, names["pi"], "pi")
    change = _gap_input(frame, names, gap).diff(_ppy(frame.index) if lag is None else lag)
    return rate.shift(1) + a_pi * (pi - target) + a_gap * change


# --- Panel -------------------------------------------------------------------


RULES: dict[str, Callable[..., pd.Series]] = {
    "taylor93": taylor1993,
    "balanced": balanced_approach,
    "shortfalls": balanced_approach_shortfalls,
    "inertial": inertial,
    "first_diff": first_difference,
}


def prescriptions(
    frame: pd.DataFrame,
    r_star: RStar = 2.0,
    *,
    rules: Mapping[str, Callable[..., pd.Series]] | None = None,
    elb: float | None = ELB,
    keep_unfloored: bool = True,
    **kw: object,
) -> pd.DataFrame:
    """Run every rule over a frame and floor the prescriptions at the ELB.

    Args:
        frame: Analysis frame.
        r_star: Neutral real rate in percent, constant or series.
        rules: Name-to-rule mapping; RULES by default.
        elb: Floor in percent; None leaves prescriptions unfloored.
        keep_unfloored: Add rule_*_unfloored columns when a floor is applied.
        **kw: Forwarded to each rule, filtered to its signature.

    Returns:
        Wide frame of ``rule_*`` columns on the frame index; with a floor and
        ``keep_unfloored``, each floored column is paired with a
        ``rule_*_unfloored`` column of unclipped values.
    """
    out: dict[str, pd.Series] = {}
    for name, fn in (rules or RULES).items():
        params = inspect.signature(fn).parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        call: dict[str, object] = {k: v for k, v in kw.items() if var_kw or k in params}
        if var_kw or "r_star" in params:
            call["r_star"] = r_star
        raw = fn(frame, **call)
        col = f"rule_{name}"
        out[col] = raw if elb is None else raw.clip(lower=elb)
        if keep_unfloored and elb is not None:
            out[f"{col}_unfloored"] = raw
    return pd.DataFrame(out, index=frame.index)
