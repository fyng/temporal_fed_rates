"""Tests for fedrates.transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from fedrates.transforms import (
    _ppy,
    align,
    annualised,
    inflation_gap,
    output_gap,
    quarterly_to_monthly,
    real_rate,
    splice,
    term_spread,
    to_monthly,
    to_quarterly,
    unemployment_gap,
    yoy,
)


def _ms(values: Sequence[float], start: str = "2024-01-01") -> pd.Series:
    """Monthly series on a month-start index named ``date``."""
    idx = pd.date_range(start, periods=len(values), freq="MS").rename("date")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _geo(growth: float, n: int, start: str, freq: str) -> pd.Series:
    """Series growing ``growth`` per period from a base of 100."""
    idx = pd.date_range(start, periods=n, freq=freq).rename("date")
    return pd.Series(100 * growth ** np.arange(n), index=idx)


# --- Frequency conversion ----------------------------------------------------


def test_to_monthly_mean_and_last():
    s = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.DatetimeIndex(
            ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
        ).rename("date"),
    )
    mean = to_monthly(s)
    assert list(mean.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert mean.index.name == "date"
    assert list(mean) == [2.0, 4.5]
    assert list(to_monthly(s, "last")) == [3.0, 5.0]


def test_to_monthly_min_obs_drops_partial_month():
    s = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.DatetimeIndex(
            ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
        ).rename("date"),
    )
    partial = to_monthly(s, "mean", min_obs=3)
    assert partial.iloc[0] == 2.0
    assert pd.isna(partial.iloc[1])
    assert list(to_monthly(s, "mean", min_obs=1)) == [2.0, 4.5]


def test_to_quarterly_mean_last_and_min_obs():
    s = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.DatetimeIndex(["2024-03-29", "2024-03-30", "2024-04-01", "2024-04-02"]).rename(
            "date"
        ),
    )
    mean = to_quarterly(s)
    assert list(mean.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")]
    assert mean.index.name == "date"
    assert list(mean) == [1.5, 3.5]
    assert list(to_quarterly(s, "last")) == [2.0, 4.0]
    guarded = to_quarterly(s, "mean", min_obs=3)
    assert pd.isna(guarded.iloc[0])
    assert pd.isna(guarded.iloc[1])


def test_quarterly_to_monthly_ffill_dates_and_values():
    qs = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]).rename(
            "date"
        ),
    )
    out = quarterly_to_monthly(qs)
    assert list(out.index) == [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 13)]
    assert list(out) == [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3
    interp = quarterly_to_monthly(qs, "interpolate")
    assert interp.iloc[1] == pytest.approx(1 + 1 / 3)
    assert interp.iloc[2] == pytest.approx(1 + 2 / 3)
    with pytest.raises(ValueError, match="ffill"):
        quarterly_to_monthly(qs, "mean")


# --- Rates of change ---------------------------------------------------------


def test_yoy_monthly_geometric_exact_rate():
    s = _geo(1.01, 25, "2022-01-01", "MS")
    out = yoy(s)
    assert out.iloc[:12].isna().all()
    assert out.iloc[12] == pytest.approx(12.6825, abs=1e-3)
    assert out.iloc[24] == pytest.approx(12.6825, abs=1e-3)


def test_yoy_log_differs_predictably():
    s = _geo(1.01, 25, "2022-01-01", "MS")
    arith = yoy(s).iloc[12]
    logs = yoy(s, log=True).iloc[12]
    assert logs == pytest.approx(100 * 12 * np.log(1.01), rel=1e-9)
    assert logs == pytest.approx(11.9404, abs=1e-3)
    assert logs < arith


def test_yoy_infers_quarterly_periods():
    s = _geo(1.02, 9, "2022-01-01", "QS")
    assert yoy(s).iloc[4] == pytest.approx(100 * 1.02**4 - 100, rel=1e-9)
    assert yoy(s).iloc[4] == yoy(s, periods=4).iloc[4]


def test_annualised_monthly_periods_three():
    s = _geo(1.01, 25, "2022-01-01", "MS")
    out = annualised(s, periods=3)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx(100 * 1.01**12 - 100, rel=1e-9)


def test_annualised_quarterly_and_explicit_ppy():
    s = _geo(1.02, 9, "2022-01-01", "QS")
    assert annualised(s, periods=1).iloc[1] == pytest.approx(100 * 1.02**4 - 100, rel=1e-9)
    q12 = annualised(s, periods=4, ppy=12)
    assert q12.iloc[4] == pytest.approx(100 * 1.02**12 - 100, rel=1e-9)


def test_ppy_monthly_quarterly_and_error():
    assert _ppy(pd.date_range("2024-01-01", periods=15, freq="MS")) == 12
    assert _ppy(pd.date_range("2024-01-01", periods=6, freq="QS")) == 4
    assert _ppy(pd.DatetimeIndex(["2024-01-01", "2024-02-01"])) == 12
    assert _ppy(pd.DatetimeIndex(["2024-01-01", "2024-04-01"])) == 4
    with pytest.raises(ValueError):
        _ppy(pd.date_range("2024-01-01", periods=15, freq="D"))
    with pytest.raises(ValueError):
        _ppy(pd.DatetimeIndex(["2024-01-01"]))


# --- Alignment ---------------------------------------------------------------


def test_align_ragged_ranges_per_key_how_and_order():
    a = _ms([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "2024-01-01")
    b = _ms([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], "2024-03-01")
    c = pd.Series(
        [5.0, 7.0], index=pd.DatetimeIndex(["2024-01-01", "2024-04-01"]).rename("date")
    )
    out = align({"a": a, "b": b, "c": c}, how={"c": "ffill"})
    assert out.index.name == "date"
    assert list(out.index) == [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 9)]
    assert list(out.columns) == ["a", "b", "c"]
    assert list(out["a"].isna()) == [False] * 6 + [True] * 2
    assert list(out["b"].isna()) == [True] * 2 + [False] * 6
    assert list(out["c"])[:6] == [5.0, 5.0, 5.0, 7.0, 7.0, 7.0]
    assert out["c"].iloc[6:].isna().all()


def test_align_trim_and_quarterly_freq():
    a = _ms([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "2024-01-01")
    out = align({"a": a}, start="2024-02-01", end="2024-04-01")
    assert list(out.index) == [
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-04-01"),
    ]
    q = align({"a": a}, freq="q")
    assert q.index.name == "date"
    assert list(q.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")]
    assert list(q["a"]) == [2.0, 5.0]
    with pytest.raises(ValueError):
        align({"a": a}, freq="d")


def test_splice_at_switches_on_date():
    primary = _ms([float(v) for v in range(1, 13)], "2024-01-01")
    fallback = _ms([float(v) for v in range(100, 112)], "2024-01-01")
    out = splice(primary, fallback, at="2024-07-01")
    assert list(out)[:6] == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert list(out)[6:] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_splice_where_and_union_index():
    primary = _ms([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "2024-01-01")
    fallback = _ms([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], "2024-02-01")
    where = pd.Series([True, True], index=pd.DatetimeIndex(["2024-02-01", "2024-05-01"]))
    out = splice(primary, fallback, where=where)
    assert list(out.index) == [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 8)]
    assert list(out)[:6] == [1.0, 20.0, 3.0, 4.0, 23.0, 6.0]
    assert pd.isna(out.iloc[6])


def test_splice_requires_exactly_one_selector():
    primary = _ms([1.0, 2.0], "2024-01-01")
    fallback = _ms([3.0, 4.0], "2024-01-01")
    with pytest.raises(ValueError):
        splice(primary, fallback)
    with pytest.raises(ValueError):
        splice(
            primary,
            fallback,
            at="2024-02-01",
            where=pd.Series([True, False], index=primary.index),
        )


# --- Gaps and spreads --------------------------------------------------------


def test_unemployment_gap_sign_and_value():
    u = _ms([6.0, 4.0], "2024-01-01")
    ustar = _ms([4.5, 4.5], "2024-01-01")
    gap = unemployment_gap(u, ustar)
    assert list(gap) == [1.5, -0.5]
    assert gap.iloc[0] > 0
    assert gap.iloc[1] < 0


def test_output_gap_sign_and_exact_value():
    y = _ms([110.0, 90.0], "2024-01-01")
    ystar = _ms([100.0, 100.0], "2024-01-01")
    gap = output_gap(y, ystar)
    assert gap.iloc[0] == pytest.approx(100 * np.log(1.1), rel=1e-12)
    assert gap.iloc[0] == pytest.approx(9.531, abs=1e-3)
    assert gap.iloc[1] == pytest.approx(100 * np.log(0.9), rel=1e-12)
    assert gap.iloc[0] > 0 > gap.iloc[1]


def test_inflation_gap():
    s = _geo(1.01, 9, "2022-01-01", "QS")
    out = inflation_gap(s)
    assert out.iloc[4] == pytest.approx(100 * 1.01**4 - 100 - 2.0, rel=1e-9)
    custom = inflation_gap(s, target=0.0, periods=4)
    assert custom.iloc[4] == pytest.approx(100 * 1.01**4 - 100, rel=1e-9)


def test_real_rate_and_term_spread_aligned():
    nominal = _ms([5.0, 3.0], "2024-01-01")
    inflation = _ms([2.0, 4.0], "2024-01-01")
    assert list(real_rate(nominal, inflation)) == [3.0, -1.0]
    long_r = _ms([4.0, 3.0], "2024-01-01")
    short_r = _ms([2.0, 5.0], "2024-01-01")
    assert list(term_spread(long_r, short_r)) == [2.0, -2.0]


def test_real_rate_misaligned_inputs():
    nominal = _ms([5.0, 3.0, 2.0], "2024-01-01")
    inflation = _ms([2.0, 1.0], "2024-01-01")
    out = real_rate(nominal, inflation)
    assert list(out.index) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]
    assert list(out)[:2] == [3.0, 2.0]
    assert pd.isna(out.iloc[2])


def test_term_spread_misaligned_inputs():
    long_r = _ms([4.0, 4.5], "2024-02-01")
    short_r = _ms([2.0, 3.0, 1.0], "2024-01-01")
    out = term_spread(long_r, short_r)
    assert pd.isna(out.iloc[0])
    assert list(out)[1:] == [1.0, 3.5]
