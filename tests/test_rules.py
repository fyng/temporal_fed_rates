"""Tests for fedrates.rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedrates.rules import (
    ELB,
    RULES,
    balanced_approach,
    balanced_approach_shortfalls,
    first_difference,
    inertial,
    prescriptions,
    taylor1993,
)

# Hand-computed scalar point on default column names: r*=2, pi=3, target=2,
# u_gap=-1 (hot), y_gap=1, observed rate 4.
SCALAR = {"pce_core_yoy": 3.0, "u_gap": -1.0, "y_gap": 1.0, "policy_rate": 4.0}


def _frame(
    columns: dict[str, float | list[float]],
    *,
    n: int | None = None,
    freq: str = "MS",
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """Frame from scalar or per-row columns on a date-named index."""
    if n is None:
        n = max((len(v) for v in columns.values() if isinstance(v, list)), default=1)
    idx = pd.date_range(start, periods=n, freq=freq).rename("date")
    data = {k: (v if isinstance(v, list) else [v] * n) for k, v in columns.items()}
    return pd.DataFrame(data, index=idx)


# --- Scalar arithmetic -------------------------------------------------------


def test_taylor_and_balanced_unemployment_form():
    f = _frame(SCALAR)
    assert taylor1993(f).iloc[0] == pytest.approx(6.50)
    assert balanced_approach(f).iloc[0] == pytest.approx(7.50)


def test_taylor_and_balanced_output_form():
    f = _frame(SCALAR)
    assert taylor1993(f, gap="output").iloc[0] == pytest.approx(6.00)
    assert balanced_approach(f, gap="output").iloc[0] == pytest.approx(6.50)


def test_shortfalls_slack_matches_balanced_and_keeps_inflation_term():
    slack = _frame({"pce_core_yoy": 3.0, "u_gap": 1.0, "y_gap": 1.0})
    s = balanced_approach_shortfalls(slack).iloc[0]
    assert s == pytest.approx(balanced_approach(slack).iloc[0])
    assert s == pytest.approx(3.5)
    hotter = _frame({"pce_core_yoy": 5.0, "u_gap": 1.0, "y_gap": 1.0})
    assert balanced_approach_shortfalls(hotter).iloc[0] == pytest.approx(s + 3.0)


def test_shortfalls_hot_drops_gap_term_and_keeps_inflation_term():
    hot = _frame({"pce_core_yoy": 3.0, "u_gap": -1.0, "y_gap": 1.0})
    s = balanced_approach_shortfalls(hot).iloc[0]
    assert s == pytest.approx(5.5)
    assert s < balanced_approach(hot).iloc[0]
    hotter = _frame({"pce_core_yoy": 5.0, "u_gap": -1.0, "y_gap": 1.0})
    assert balanced_approach_shortfalls(hotter).iloc[0] == pytest.approx(s + 3.0)


def test_a_gap_none_resolves_default_per_rule_and_gap():
    f = _frame(SCALAR)
    assert taylor1993(f).iloc[0] == pytest.approx(taylor1993(f, a_gap=1.0).iloc[0])
    assert balanced_approach(f).iloc[0] == pytest.approx(balanced_approach(f, a_gap=2.0).iloc[0])
    assert taylor1993(f, gap="output").iloc[0] == pytest.approx(
        taylor1993(f, gap="output", a_gap=0.5).iloc[0]
    )
    assert balanced_approach(f, gap="output").iloc[0] == pytest.approx(
        balanced_approach(f, gap="output", a_gap=1.0).iloc[0]
    )
    assert taylor1993(f).iloc[0] != pytest.approx(taylor1993(f, a_gap=2.0).iloc[0])
    assert balanced_approach(f).iloc[0] != pytest.approx(balanced_approach(f, a_gap=1.0).iloc[0])


# --- r* broadcasting ---------------------------------------------------------


def test_r_star_series_broadcasts_elementwise():
    f = _frame({"pce_core_yoy": [3.0, 3.0], "u_gap": -1.0, "y_gap": 1.0})
    r_star = pd.Series([1.0, 3.0], index=f.index)
    out = taylor1993(f, r_star)
    assert out.iloc[0] == pytest.approx(5.5)
    assert out.iloc[1] == pytest.approx(7.5)


def test_r_star_shorter_series_reindexes_to_nan():
    f = _frame({"pce_core_yoy": [3.0, 3.0], "u_gap": -1.0, "y_gap": 1.0})
    r_star = pd.Series([1.0], index=f.index[:1])
    out = taylor1993(f, r_star)
    assert out.iloc[0] == pytest.approx(5.5)
    assert pd.isna(out.iloc[1])


# --- Inertial rule -----------------------------------------------------------


def test_inertial_actual_uses_lagged_observed_rate():
    f = _frame(SCALAR, n=13)
    out = inertial(f, mode="actual")
    assert pd.isna(out.iloc[0])
    assert np.allclose(out.iloc[1:], 0.85 * 4.0 + 0.15 * 7.50)


def test_inertial_recursive_seeded_and_diverges_from_actual():
    f = _frame(SCALAR, n=13)
    rec = inertial(f, mode="recursive")
    act = inertial(f, mode="actual")
    assert rec.iloc[0] == pytest.approx(4.0)
    assert rec.notna().all()
    assert rec.iloc[1] == pytest.approx(0.85 * 4.0 + 0.15 * 7.5)
    assert rec.iloc[2] == pytest.approx(0.85 * rec.iloc[1] + 0.15 * 7.5)
    assert act.iloc[2] == pytest.approx(4.525)
    assert rec.iloc[2] != pytest.approx(act.iloc[2])
    assert rec.iloc[-1] > rec.iloc[1]


def test_inertial_recursive_holds_through_nan_base_values():
    f = _frame(
        {
            "pce_core_yoy": [3.0, np.nan, np.nan, 3.0],
            "u_gap": -1.0,
            "y_gap": 1.0,
            "policy_rate": 4.0,
        }
    )
    out = inertial(f, mode="recursive")
    assert out.notna().all()
    assert out.iloc[1] == out.iloc[2] == pytest.approx(4.0)
    assert out.iloc[3] == pytest.approx(0.85 * 4.0 + 0.15 * 7.5)


def test_inertial_bad_mode_raises():
    f = _frame(SCALAR)
    with pytest.raises(ValueError, match="mode"):
        inertial(f, mode="bogus")


# --- First-difference rule ---------------------------------------------------


def test_first_difference_hand_computed_monthly():
    f = _frame(
        {"pce_core_yoy": 3.0, "u_gap": [-0.5] + [-1.0] * 12, "y_gap": 1.0, "policy_rate": 4.0}
    )
    out = first_difference(f)
    assert out.iloc[:12].isna().all()
    assert out.iloc[-1] == pytest.approx(5.00)
    assert out.iloc[-1] == pytest.approx(first_difference(f, lag=12).iloc[-1])
    assert out.iloc[-1] != pytest.approx(first_difference(f, lag=1).iloc[-1])
    assert first_difference(f, gap="output").iloc[-1] == pytest.approx(4.50)


def test_first_difference_infers_quarterly_lag():
    f = _frame(
        {"pce_core_yoy": 3.0, "u_gap": [-0.5] + [-1.0] * 4, "y_gap": 1.0, "policy_rate": 4.0},
        freq="QS",
    )
    assert first_difference(f).iloc[-1] == pytest.approx(5.00)


# --- Column remapping --------------------------------------------------------


def test_cols_remap_works_for_all_inputs():
    cols = {"pi": "cpi_core", "u_gap": "ugap", "y_gap": "outputgap", "rate": "ffr"}
    f = _frame({"cpi_core": 3.0, "ugap": -1.0, "outputgap": 1.0, "ffr": 4.0})
    assert balanced_approach(f, cols=cols).iloc[0] == pytest.approx(7.50)
    f13 = _frame({"cpi_core": 3.0, "ugap": [-0.5] + [-1.0] * 12, "outputgap": 1.0, "ffr": 4.0})
    assert first_difference(f13, cols=cols).iloc[-1] == pytest.approx(5.00)
    assert inertial(f13, cols=cols).iloc[-1] == pytest.approx(4.525)


def test_missing_column_raises_keyerror_naming_it():
    f = _frame({"u_gap": -1.0, "y_gap": 1.0, "policy_rate": 4.0})
    with pytest.raises(KeyError, match="pce_core_yoy"):
        taylor1993(f)
    g = _frame({"pce_core_yoy": 3.0, "u_gap": -1.0, "y_gap": 1.0})
    with pytest.raises(KeyError, match="policy_rate"):
        first_difference(g)
    with pytest.raises(KeyError, match="policy_rate"):
        inertial(g, mode="recursive")


def test_bad_gap_and_mode_raise_valueerror():
    f = _frame(SCALAR)
    with pytest.raises(ValueError, match="gap"):
        taylor1993(f, gap="bogus")
    with pytest.raises(ValueError, match="gap"):
        first_difference(f, gap="bogus")


# --- ELB and prescriptions panel ---------------------------------------------


def test_rules_mapping_and_elb_constant():
    assert list(RULES) == ["taylor93", "balanced", "shortfalls", "inertial", "first_diff"]
    assert ELB == 0.125


def test_prescriptions_columns_index_and_values(synthetic_monthly):
    out = prescriptions(synthetic_monthly)
    names = list(RULES)
    floored = [f"rule_{n}" for n in names]
    unfloored = [f"{c}_unfloored" for c in floored]
    assert set(out.columns) == set(floored) | set(unfloored)
    assert list(out[floored].columns) == floored
    assert out.index.equals(synthetic_monthly.index)
    assert out.index.name == "date"
    for n in names:
        raw = RULES[n](synthetic_monthly)
        pd.testing.assert_series_equal(out[f"rule_{n}_unfloored"], raw, check_names=False)
        above = raw >= ELB
        assert out.loc[above, f"rule_{n}"].equals(raw[above])
        below = (~above) & raw.notna()
        assert (out.loc[below, f"rule_{n}"] == ELB).all()


def test_prescriptions_floor_clips_and_keeps_unfloored():
    f = _frame({"pce_core_yoy": 0.0, "u_gap": 5.0, "y_gap": -1.0, "policy_rate": 0.125}, n=13)
    out = prescriptions(f)
    mains = out[[f"rule_{n}" for n in RULES]]
    unfloored = out[[f"rule_{n}_unfloored" for n in RULES]]
    assert mains.iloc[-1].eq(ELB).all()
    assert (unfloored.iloc[-1] < 0).all()
    assert out["rule_balanced_unfloored"].iloc[-1] == pytest.approx(-9.0)
    assert out["rule_first_diff_unfloored"].iloc[-1] == pytest.approx(-0.875)


def test_prescriptions_elb_none_skips_unfloored_columns():
    f = _frame(SCALAR, n=13)
    out = prescriptions(f, elb=None)
    assert list(out.columns) == [f"rule_{n}" for n in RULES]
    for n in RULES:
        pd.testing.assert_series_equal(out[f"rule_{n}"], RULES[n](f), check_names=False)


def test_prescriptions_forwards_kw_only_where_accepted():
    f = _frame(SCALAR, n=13)
    out = prescriptions(f, gap="output", elb=None)
    assert out["rule_taylor93"].iloc[-1] == pytest.approx(6.00)
    assert out["rule_balanced"].iloc[-1] == pytest.approx(6.50)
    assert out["rule_shortfalls"].iloc[-1] == pytest.approx(5.5)
    assert out["rule_inertial"].iloc[-1] == pytest.approx(0.85 * 4.0 + 0.15 * 6.50)
    assert out["rule_first_diff"].iloc[-1] == pytest.approx(4.50)


# --- Sign sanity -------------------------------------------------------------


def test_hot_economy_prescribes_above_neutral():
    hot = _frame(
        {"pce_core_yoy": 4.0, "u_gap": -1.0, "y_gap": 1.0, "policy_rate": 4.0}, n=13
    )
    for name, fn in RULES.items():
        assert fn(hot).iloc[-1] > 2.0 + 2.0, name


def test_slack_economy_prescribes_below_neutral():
    slack = _frame(
        {"pce_core_yoy": 0.0, "u_gap": 3.0, "y_gap": -1.0, "policy_rate": 4.0}, n=13
    )
    for name, fn in RULES.items():
        assert fn(slack).iloc[-1] < 2.0 + 2.0, name
