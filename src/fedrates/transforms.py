"""Frequency, alignment and gap transforms for FRED series.

All outputs carry a ``DatetimeIndex`` named ``date``. Sign conventions:
``unemployment_gap`` is u − u* (positive means slack); ``output_gap`` is
100·log(y/y*) (positive means above potential).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "to_monthly",
    "to_quarterly",
    "quarterly_to_monthly",
    "align",
    "splice",
    "yoy",
    "annualised",
    "inflation_gap",
    "unemployment_gap",
    "output_gap",
    "real_rate",
    "term_spread",
]

Aggregation = Literal["mean", "last", "sum", "ffill"]

_AGGREGATIONS = ("mean", "last", "sum", "ffill")
_FREQ = {"m": "MS", "q": "QS"}


# --- Index helpers -----------------------------------------------------------


def _named(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Rename a newly created series or frame's index to ``date``.

    Args:
        s: Series or frame to rename; mutated in place.

    Returns:
        The input with ``index.name == "date"``.
    """
    s.index = s.index.rename("date")
    return s


def _ppy_value(index: pd.DatetimeIndex) -> int | None:
    """Periods per year for an index.

    Args:
        index: Index to inspect.

    Returns:
        12 for monthly, 4 for quarterly, None otherwise.
    """
    freq = None
    if len(index) >= 3:
        try:
            freq = pd.infer_freq(index)
        except (TypeError, ValueError):
            freq = None
    if freq is None:
        if len(index) < 2:
            return None
        days = float(np.median((index[1:] - index[:-1]) / pd.Timedelta(days=1)))
        if 20 <= days <= 45:
            return 12
        if 75 <= days <= 110:
            return 4
        return None
    first = freq[0].upper()
    if first == "M":
        return 12
    if first == "Q":
        return 4
    return None


def _ppy(index: pd.DatetimeIndex) -> int:
    """Periods per year for a monthly or quarterly index.

    Args:
        index: Index to inspect.

    Returns:
        12 for monthly, 4 for quarterly.

    Raises:
        ValueError: If the frequency is neither monthly nor quarterly.
    """
    ppy = _ppy_value(index)
    if ppy is None:
        raise ValueError(
            f"cannot infer periods per year from {len(index)} timestamps: "
            "expected a monthly or quarterly index"
        )
    return ppy


def _spread_quarterly(s: pd.Series, interpolate: bool) -> pd.Series:
    """Reindex a quarter-start series to month-start, filling each quarter.

    Args:
        s: Series on a QS index.
        interpolate: Interpolate between quarter points instead of repeating;
            months past the last observation repeat it.

    Returns:
        Series on an MS index.

    Raises:
        ValueError: If the series is empty.
    """
    if s.empty:
        raise ValueError("cannot spread an empty quarterly series")
    idx = pd.date_range(s.index[0], s.index[-1] + pd.DateOffset(months=2), freq="MS")
    out = s.reindex(idx)
    return out.interpolate() if interpolate else out.ffill()


def _collapse(s: pd.Series, freq: str, how: Aggregation, min_obs: int) -> pd.Series:
    """Resample to ``freq`` with an aggregation.

    Args:
        s: Series with a DatetimeIndex.
        freq: Pandas offset alias, "MS" or "QS".
        how: Aggregation; "ffill" carries the last value forward, the others
            aggregate each bin.
        min_obs: Minimum observations per bin; fewer yields NaN.

    Returns:
        Resampled series.

    Raises:
        ValueError: If ``how`` is not a supported aggregation.
    """
    if how not in _AGGREGATIONS:
        raise ValueError(f"how must be one of {_AGGREGATIONS}, got {how!r}")
    if how == "ffill":
        if freq == "MS" and len(s) > 1 and _ppy_value(s.index) == 4:
            return _spread_quarterly(s, interpolate=False)
        return s.resample(freq).ffill()
    out = s.resample(freq).agg(how)
    return out.where(s.resample(freq).count() >= min_obs)


# --- Frequency and alignment -------------------------------------------------


def to_monthly(s: pd.Series, how: Aggregation = "mean", min_obs: int = 1) -> pd.Series:
    """Collapse a series to month-start frequency.

    Args:
        s: Series with a DatetimeIndex.
        how: Aggregation over each month.
        min_obs: Minimum observations per month; fewer yields NaN. Guards the
            partial current month when resampling a daily series mid-month.

    Returns:
        Series on an MS index named ``date``.
    """
    return _named(_collapse(s, "MS", how, min_obs))


def to_quarterly(s: pd.Series, how: Aggregation = "mean", min_obs: int = 1) -> pd.Series:
    """Collapse a series to quarter-start frequency.

    Args:
        s: Series with a DatetimeIndex.
        how: Aggregation over each quarter.
        min_obs: Minimum observations per quarter; fewer yields NaN.

    Returns:
        Series on a QS index named ``date``.
    """
    return _named(_collapse(s, "QS", how, min_obs))


def quarterly_to_monthly(
    s: pd.Series, how: Literal["ffill", "interpolate"] = "ffill"
) -> pd.Series:
    """Spread a quarter-start series over the three months of each quarter.

    Args:
        s: Series on a QS index.
        how: "ffill" repeats each quarterly value; "interpolate" interpolates
            linearly between quarter points.

    Returns:
        Series on an MS index named ``date``.

    Raises:
        ValueError: If ``how`` is not "ffill" or "interpolate".
    """
    if how not in ("ffill", "interpolate"):
        raise ValueError(f"how must be 'ffill' or 'interpolate', got {how!r}")
    return _named(_spread_quarterly(s, interpolate=how == "interpolate"))


def align(
    series: Mapping[str, pd.Series],
    *,
    freq: Literal["m", "q"] = "m",
    how: Mapping[str, Aggregation] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Outer-join series on a common frequency.

    Args:
        series: Named series; column order follows the mapping's insertion
            order.
        freq: Target frequency, "m" for month-start or "q" for quarter-start.
        how: Per-series aggregation, defaulting to "mean".
        start: Inclusive start of the output range.
        end: Inclusive end of the output range.

    Returns:
        DataFrame on an MS or QS index named ``date``.

    Raises:
        ValueError: If ``freq`` is not "m" or "q".
    """
    if freq not in _FREQ:
        raise ValueError(f"freq must be 'm' or 'q', got {freq!r}")
    how = how or {}
    cols = {
        name: _collapse(s, _FREQ[freq], how.get(name, "mean"), 1) for name, s in series.items()
    }
    out = pd.DataFrame(cols)
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    if lo is not None or hi is not None:
        out = out.loc[lo:hi]
    return _named(out)


def splice(
    primary: pd.Series,
    fallback: pd.Series,
    *,
    at: str | pd.Timestamp | None = None,
    where: pd.Series | None = None,
) -> pd.Series:
    """Replace parts of ``primary`` with values from ``fallback``.

    Args:
        primary: Base series.
        fallback: Series supplying the spliced values.
        at: Cutover timestamp; ``fallback`` supplies values strictly before it.
        where: Boolean mask; True positions take ``fallback``.

    Returns:
        Series on the union index, named ``date``.

    Raises:
        ValueError: If not exactly one of ``at`` and ``where`` is given.
    """
    if (at is None) == (where is None):
        raise ValueError("pass exactly one of at= or where=")
    idx = primary.index.union(fallback.index)
    p = primary.reindex(idx)
    f = fallback.reindex(idx)
    if at is not None:
        mask = pd.Series(idx < pd.Timestamp(at), index=idx)
    else:
        mask = where.reindex(idx).eq(True)
    return _named(f.where(mask, p))


# --- Rates of change ---------------------------------------------------------


def yoy(s: pd.Series, periods: int | None = None, *, log: bool = False) -> pd.Series:
    """Year-on-year percent change.

    Args:
        s: Series with a DatetimeIndex.
        periods: Lag length; inferred from the index (12 monthly, 4 quarterly)
            when None.
        log: Take differences of logs instead of arithmetic growth.

    Returns:
        Percent change; 3.0 means 3%.
    """
    periods = _ppy(s.index) if periods is None else periods
    if log:
        return _named(100 * (np.log(s) - np.log(s.shift(periods))))
    return _named(100 * (s / s.shift(periods) - 1))


def annualised(s: pd.Series, periods: int = 3, ppy: int | None = None) -> pd.Series:
    """Compound-annualised percent rate of change over ``periods`` periods.

    Args:
        s: Series with a DatetimeIndex.
        periods: Number of periods of change to annualise.
        ppy: Periods per year; inferred from the index when None.

    Returns:
        Annualised percent rate; 3.0 means 3%.
    """
    ppy = _ppy(s.index) if ppy is None else ppy
    return _named(100 * ((s / s.shift(periods)) ** (ppy / periods) - 1))


# --- Gaps and spreads --------------------------------------------------------


def inflation_gap(
    price_index: pd.Series, *, target: float = 2.0, periods: int | None = None
) -> pd.Series:
    """Inflation minus its target, in percentage points.

    Args:
        price_index: Price level series.
        target: Inflation target in percent.
        periods: Lag length for the year-on-year computation; inferred from
            the index when None.

    Returns:
        Year-on-year inflation in percent minus ``target``.
    """
    return yoy(price_index, periods=periods) - target


def unemployment_gap(unrate: pd.Series, nrou: pd.Series) -> pd.Series:
    """Unemployment rate minus its natural rate.

    Args:
        unrate: Unemployment rate.
        nrou: Natural rate of unemployment.

    Returns:
        u − u*; positive means slack.
    """
    return _named(unrate - nrou)


def output_gap(gdp: pd.Series, potential: pd.Series) -> pd.Series:
    """Percent log deviation of output from potential.

    Args:
        gdp: Real output.
        potential: Potential output.

    Returns:
        100·log(y / y*); positive means above potential.
    """
    return _named(100 * np.log(gdp / potential))


def real_rate(nominal: pd.Series, inflation: pd.Series) -> pd.Series:
    """Ex-post real interest rate.

    Args:
        nominal: Nominal rate in percent.
        inflation: Inflation rate in percent.

    Returns:
        nominal − inflation.
    """
    return _named(nominal - inflation)


def term_spread(long_rate: pd.Series, short_rate: pd.Series) -> pd.Series:
    """Long rate minus short rate, in percentage points.

    Args:
        long_rate: Long-maturity yield.
        short_rate: Short-maturity yield.

    Returns:
        long − short.
    """
    return _named(long_rate - short_rate)
