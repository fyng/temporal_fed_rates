"""Tests for fedrates.models.decision: features, ordered logit, metrics, walk-forward."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fedrates import fomc
from fedrates.models import decision
from fedrates.models.decision import (
    CLASSES,
    FEATURES,
    HISTORY,
    MACRO,
    MARKET,
    RULE_GAPS,
    SPECS,
    ablation,
    cut25_diagnosis,
    evaluate,
    features,
    fit_ordered_logit,
    headline,
    labels,
    predict_proba,
    quadratic_weighted_kappa,
    rps,
    walk_forward,
)

# --- Synthetic data helpers ---------------------------------------------------

_FRAME_COLS = (
    "pi_gap", "u_gap", "y_gap", "term_spread", "nfci", "vix", "claims", "bei10",
    "mich", "r_star", "payems", "rule_taylor93", "rule_balanced", "rule_shortfalls",
    "rule_inertial", "rule_first_diff", "target_old", "target_upper", "target_lower",
    "recession",
)
_SIZES = (-50, -25, 0, 0, 25, 50, 0, -25, 0, 25, 0, -50)


def _meeting_dates(n: int) -> pd.DatetimeIndex:
    """n mid-month meeting dates, one per month from 2022-01."""
    return pd.DatetimeIndex(
        pd.date_range("2022-01-01", periods=n, freq="MS") + pd.Timedelta(days=14), name="date"
    )


def _month_starts(dates: pd.DatetimeIndex, warmup: int = 3) -> pd.DatetimeIndex:
    """Month-start frame index covering the meetings plus ``warmup`` lead months."""
    first = pd.Timestamp(dates.min()).to_period("M").to_timestamp() - pd.DateOffset(months=warmup)
    last = pd.Timestamp(dates.max()).to_period("M").to_timestamp()
    return pd.date_range(first, last, freq="MS")


def _monthly_frame(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic frame: column j of row i carries 100 * i + j."""
    return pd.DataFrame(
        {col: 100.0 * np.arange(len(months)) + j for j, col in enumerate(_FRAME_COLS)},
        index=pd.DatetimeIndex(months, name="date"),
    )


def _meeting_frame(dates: pd.DatetimeIndex, sizes, intermeeting=()) -> pd.DataFrame:
    """Meetings with single-point targets stepping 25bp per decision."""
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "size_bp": list(sizes),
            "intermeeting": [d in set(intermeeting) for d in dates],
            "target": 3.0 + 0.25 * np.arange(n),
            "target_lower": np.full(n, np.nan),
            "target_upper": np.full(n, np.nan),
        }
    )


def _cyclic_xy(n: int) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic X, y hitting all five classes, index dated monthly."""
    codes = np.arange(n) % len(CLASSES)
    y = pd.Series(
        pd.Categorical.from_codes(codes, list(CLASSES), ordered=True),
        index=pd.date_range("2020-01-01", periods=n, freq="MS", name="date"),
    )
    X = pd.DataFrame(
        {"a": (codes - 2).astype(float), "b": np.sin(np.arange(n, dtype=float))}, index=y.index
    )
    return X, y


# --- Features -----------------------------------------------------------------


def test_features_one_row_per_meeting_aligned() -> None:
    """The design matrix is indexed by meeting date with FEATURES plus MARKET."""
    dates = _meeting_dates(4)
    X = features(_monthly_frame(_month_starts(dates)), _meeting_frame(dates, (-25, 0, 25, 50)))
    assert list(X.index) == list(dates)
    assert X.index.name == "date"
    assert list(X.columns) == [*FEATURES, *MARKET]


def test_features_read_exactly_the_lagged_months() -> None:
    """Level features read the lag-1 frame row, payroll change the lag-2 row.

    Meeting k sits in frame row warmup + k, so pi_gap must be the frame value
    100 * (warmup + k - 1) and payems_chg1 the 100.0 difference between the
    lag-1 and lag-2 payems values.
    """
    dates = _meeting_dates(4)
    frame = _monthly_frame(_month_starts(dates, warmup=3))
    X = features(frame, _meeting_frame(dates, (-25, 0, 25, 50)))
    for k in range(4):
        assert X["pi_gap"].iloc[k] == pytest.approx(100.0 * (2 + k))
        assert X["mich"].iloc[k] == pytest.approx(100.0 * (2 + k) + 8)
        assert X["payems_chg1"].iloc[k] == pytest.approx(100.0)


def test_features_rule_gap_uses_previous_midpoint() -> None:
    """rule_gap = lagged prescription minus the previous decision's midpoint.

    The first meeting falls back to the frame's target_old: rule row 2 is 211
    and target_old row 2 is 216, so the gap is exactly -5.0; later meetings
    subtract the stepped synthetic targets.
    """
    dates = _meeting_dates(4)
    frame = _monthly_frame(_month_starts(dates, warmup=3))
    X = features(frame, _meeting_frame(dates, (-25, 0, 25, 50)))
    assert X["rule_gap_taylor93"].iloc[0] == pytest.approx(-5.0)
    assert X["rule_gap_taylor93"].iloc[1] == pytest.approx(311.0 - 3.0)
    assert X["rule_gap_taylor93"].iloc[2] == pytest.approx(411.0 - 3.25)


def test_features_decision_history_columns() -> None:
    """decision_lag1, mtgs_since_move and intermeeting follow the past record."""
    dates = _meeting_dates(8)
    X = features(
        _monthly_frame(_month_starts(dates)),
        _meeting_frame(dates, (-25, 0, 0, 25, 50, 0, 0, 0), intermeeting=(dates[4],)),
    )
    assert X["decision_lag1"].to_numpy()[1:] == pytest.approx([1, 2, 2, 3, 4, 2, 2])
    assert math.isnan(X["decision_lag1"].iloc[0])
    assert X["mtgs_since_move"].to_numpy() == pytest.approx([0, 1, 2, 3, 1, 1, 2, 3])
    assert X["intermeeting"].to_numpy() == pytest.approx([0, 0, 0, 0, 1, 0, 0, 0])


def test_features_never_use_meeting_month_or_later() -> None:
    """The leakage guard: poisoned meeting-month rows cannot reach any feature.

    Two meetings four months apart; every frame month from the first meeting
    month onward is poisoned with 9999.0 except the two lag months each
    meeting is allowed to read. A lag_months of 0 or negative would pull the
    poison into the matrix and fail.
    """
    dates = pd.DatetimeIndex(["2022-01-15", "2022-05-15"])
    months = pd.date_range("2021-09-01", "2022-07-01", freq="MS")
    frame = _monthly_frame(months)
    meeting_months = dates.to_period("M")
    allowed = {m - k for m in meeting_months for k in (1, 2)}
    poison = [
        t for t in months
        if t.to_period("M") >= meeting_months.min() and t.to_period("M") not in allowed
    ]
    frame.loc[poison] = 9999.0
    X = features(frame, _meeting_frame(dates, (-25, 25)))
    assert not (X == 9999.0).any().any()
    # and the values pin the lag exactly: meeting 1 reads December, meeting 2 April
    assert X["pi_gap"].iloc[0] == pytest.approx(300.0)  # frame row 2021-12
    assert X["pi_gap"].iloc[1] == pytest.approx(700.0)  # frame row 2022-04


def test_features_tbill_spread_never_uses_meeting_day_or_later() -> None:
    """The market block reads the last bill print strictly before each meeting.

    Every daily row dated on or after the first meeting date is poisoned with
    9999.0 except the single prior-business-day row each meeting may read, so
    a join on the meeting day, a later day or any monthly aggregation pulls
    poison into the spreads and fails the exact-value assertions.
    """
    dates = _meeting_dates(3)
    idx = pd.date_range("2021-11-01", "2022-04-30", freq="B", name="date")
    tbill = pd.DataFrame({"tbill3": 5.0, "tbill6": 5.5}, index=idx)
    allowed = pd.DatetimeIndex([idx[idx < d][-1] for d in dates])
    tbill.loc[idx >= dates[0]] = 9999.0
    tbill.loc[allowed, "tbill3"] = [5.1, 5.2, 5.3]
    tbill.loc[allowed, "tbill6"] = [5.4, 5.5, 5.6]
    X = features(
        _monthly_frame(_month_starts(dates)), _meeting_frame(dates, (-25, 0, 25)), tbill=tbill
    )
    # previous midpoints: frame target_old (216.0) for the first meeting, then
    # the synthetic single-point targets 3.0 and 3.25
    assert X["tbill3_spread"].to_numpy() == pytest.approx([5.1 - 216.0, 5.2 - 3.0, 5.3 - 3.25])
    assert X["tbill6_spread"].to_numpy() == pytest.approx([5.4 - 216.0, 5.5 - 3.0, 5.6 - 3.25])


def test_features_real_record() -> None:
    """The real record yields 271 rows aligned with the committed meetings."""
    X = features()
    y = labels()
    meetings = fomc.load_meetings()
    assert len(X) == len(y) == len(meetings) == 271
    assert X.index.equals(y.index)
    assert X.index.is_monotonic_increasing
    assert X["intermeeting"].sum() == float(meetings["intermeeting"].sum())


# --- Labels -------------------------------------------------------------------


def test_labels_ordered_categorical_on_meeting_dates() -> None:
    """labels returns the ordered CLASSES scale indexed by meeting date."""
    dates = _meeting_dates(len(_SIZES))
    y = labels(_meeting_frame(dates, _SIZES))
    assert list(y.cat.categories) == list(CLASSES)
    assert y.cat.ordered
    assert list(y.index) == list(dates)
    assert y.iloc[0] == "cut50+"
    assert y.iloc[2] == "hold"


def test_labels_real_record_starts_with_a_hike() -> None:
    """The full record labels 1994-02-04 (+25bp) as hike25."""
    y = labels()
    assert y.iloc[0] == "hike25"
    assert y.index[0] == pd.Timestamp("1994-02-04")


# --- Model --------------------------------------------------------------------


def test_predict_proba_rows_sum_to_one() -> None:
    """Proba has the CLASSES columns in order, in [0, 1], summing to 1."""
    X, y = _cyclic_xy(120)
    proba = predict_proba(fit_ordered_logit(X, y), X)
    assert list(proba.columns) == list(CLASSES)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba.to_numpy() >= 0).all()


def test_predict_proba_orders_the_classes() -> None:
    """A feature that drives the label monotonically orders the probabilities."""
    X, y = _cyclic_xy(120)
    X = X.assign(a=X["a"] * 3.0)
    proba = predict_proba(fit_ordered_logit(X, y), X)
    top = X["a"] >= X["a"].median()
    assert proba["hike50+"][top].mean() > proba["hike50+"][~top].mean()
    assert proba["cut50+"][~top].mean() > proba["cut50+"][top].mean()


def test_fit_drops_constant_columns_and_imputes() -> None:
    """Zero-variance columns are dropped and NaN cells are median-filled."""
    X, y = _cyclic_xy(60)
    X = X.assign(dead=1.0, hole=[np.nan] * 30 + [2.0] * 30)
    model = fit_ordered_logit(X, y)
    assert "dead" not in model.columns
    assert "hole" not in model.columns
    proba = predict_proba(model, X)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --- Metrics ------------------------------------------------------------------


def test_rps_hand_computed() -> None:
    """RPS values worked out by hand on the cumulative distribution."""
    uniform = pd.DataFrame(np.full((1, 5), 0.2), columns=list(CLASSES))
    assert rps(["hold"], uniform) == pytest.approx(0.1)
    half = np.array([[0.5, 0.5, 0.0, 0.0, 0.0]])
    assert rps(["cut50+"], half) == pytest.approx(0.0625)
    assert rps(CLASSES, np.eye(5)) == pytest.approx(0.0)
    # calling a hike50+ a cut50+ costs four times calling it a hike25
    assert rps(["hike50+"], np.array([[1.0, 0, 0, 0, 0]])) == pytest.approx(1.0)
    assert rps(["hike50+"], np.array([[0.0, 0, 0, 1, 0]])) == pytest.approx(0.25)


def test_quadratic_weighted_kappa_hand_computed() -> None:
    """Kappa worked out by hand: two exact, two adjacent errors give 0.8."""
    y_true = ["cut50+", "cut25", "hold", "hike25"]
    y_pred = ["cut50+", "cut25", "hike25", "hold"]
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(0.8)
    assert quadratic_weighted_kappa(y_true, y_true) == pytest.approx(1.0)


def test_constant_predictor_scores_kappa_zero() -> None:
    """Any constant predictor has kappa exactly 0 under the weighted score."""
    y_true = ["cut50+", "cut25", "hold", "hike25", "hike50+"]
    assert quadratic_weighted_kappa(y_true, ["hold"] * 5) == pytest.approx(0.0, abs=1e-9)


def test_evaluate_baselines() -> None:
    """Perfect predictions score 1.0 kappa and 0.0 RPS; baselines degrade."""
    y = pd.Series(
        pd.Categorical(
            ["cut50+", "cut25", "hold", "hike25"], categories=list(CLASSES), ordered=True
        )
    )
    eye = np.eye(5)[[0, 1, 2, 3]]
    proba = pd.DataFrame(eye, columns=list(CLASSES))
    res = evaluate(y, proba)
    assert res["model"]["kappa"] == pytest.approx(1.0)
    assert res["model"]["rps"] == pytest.approx(0.0)
    assert res["model"]["accuracy"] == pytest.approx(1.0)
    assert res["constant_hold"]["kappa"] == pytest.approx(0.0, abs=1e-9)
    assert math.isinf(res["constant_hold"]["log_loss"])
    assert set(res) == {"model", "expanding_prior", "prior_full_sample", "constant_hold"}
    assert res["expanding_prior"]["rps"] > res["model"]["rps"]
    assert res["prior_full_sample"]["rps"] > res["model"]["rps"]


def test_evaluate_log_loss_hand_computed() -> None:
    """Log loss is the negative log probability of the realised class.

    Two labels so the empirical prior is (0, 0.5, 0.5, 0, 0) rather than a
    point mass on hold.
    """
    y = pd.Series(
        pd.Categorical(["hold", "cut25"], categories=list(CLASSES), ordered=True)
    )
    p = pd.DataFrame(
        [[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]], columns=list(CLASSES)
    )
    res = evaluate(y, p)
    assert res["model"]["log_loss"] == pytest.approx((-math.log(0.4) - math.log(0.2)) / 2)
    assert res["prior_full_sample"]["log_loss"] == pytest.approx(math.log(2))


def test_evaluate_prior_baseline_matches_empirical_frequencies() -> None:
    """The full-sample prior baseline is the empirical distribution of y_true."""
    y = pd.Series(
        pd.Categorical(["cut25", "cut25", "cut25", "hold"], categories=list(CLASSES), ordered=True)
    )
    res = evaluate(y, np.full((4, 5), 0.2))
    assert res["prior_full_sample"]["accuracy"] == pytest.approx(0.75)
    # constant argmax prediction, so kappa is 0 by construction
    assert res["prior_full_sample"]["kappa"] == pytest.approx(0.0, abs=1e-9)
    # uniform model probabilities score 0.1375 against these labels (see below)
    assert res["model"]["rps"] == pytest.approx(0.1375)


def test_evaluate_prior_rps_hand_derivation() -> None:
    """Baseline RPS against these labels, derived threshold by threshold.

    Full-sample prior (0, 0.75, 0.25, 0, 0) has cumulative (0, 0.75, 1, 1).
    Against a cut25 truth the gaps are (0, -0.25, 0, 0) -> 0.015625; against a
    hold truth they are (0, 0.75, 0, 0) -> 0.140625. Mean over 3 cut25 and 1
    hold: 0.1875 / 4 = 0.046875. Uniform probabilities have cumulative
    (0.2, 0.4, 0.6, 0.8); a cut25 truth gives gaps squaring to 0.6 and a hold
    truth to 0.4, so the mean RPS is (3 * 0.6 + 0.4) / 16 = 0.1375.
    The expanding prior is uniform, then (0,1,0,0,0) from meeting 2 on: RPS
    rows 0.15, 0, 0, 0.25 (one-hot cut25 against a hold truth) -> mean 0.1.
    """
    y = pd.Series(
        pd.Categorical(["cut25", "cut25", "cut25", "hold"], categories=list(CLASSES), ordered=True)
    )
    res = evaluate(y, np.full((4, 5), 0.2))
    assert res["prior_full_sample"]["rps"] == pytest.approx(3 / 64)
    assert res["model"]["rps"] == pytest.approx(0.1375)
    assert res["expanding_prior"]["rps"] == pytest.approx(0.1)
    assert res["expanding_prior"]["accuracy"] == pytest.approx(0.5)


def test_evaluate_balanced_metrics_perfect_and_constant() -> None:
    """Perfect probabilities score 1 everywhere; always-hold scores chance."""
    y = pd.Series(
        pd.Categorical(
            ["cut25", "hold", "hold", "hold", "hike25", "hold"],
            categories=list(CLASSES), ordered=True,
        )
    )
    proba = np.eye(5)[np.asarray(y.cat.codes)]
    res = evaluate(y, proba)
    for key in ("auroc_macro", "auprc_macro", "f1_macro", "balanced_accuracy"):
        assert res["model"][key] == pytest.approx(1.0)
    hold = res["constant_hold"]
    assert hold["accuracy"] == pytest.approx(4 / 6)
    assert hold["auroc_macro"] == pytest.approx(0.5)
    # AP of a constant score is the class prevalence: (1/6 + 4/6 + 1/6) / 3
    assert hold["auprc_macro"] == pytest.approx(1 / 3)
    assert hold["balanced_accuracy"] == pytest.approx(1 / 3)
    # F1 is 0.8 on hold (precision 4/6, recall 1) and 0 on the two cuts/hikes
    assert hold["f1_macro"] == pytest.approx(0.8 / 3)


def test_evaluate_balanced_metrics_skip_absent_classes() -> None:
    """Classes absent from y_true score NaN and drop out of the macro average."""
    y = pd.Series(
        pd.Categorical(["hold", "cut25", "hold", "cut25"], categories=list(CLASSES), ordered=True)
    )
    p = np.array([
        [0.0, 0.2, 0.8, 0.0, 0.0],
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.4, 0.6, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0, 0.0],
    ])
    res = evaluate(y, p)["model"]
    for name in ("cut50+", "hike25", "hike50+"):
        assert math.isnan(res[f"auroc_{name}"])
        assert math.isnan(res[f"auprc_{name}"])
    assert res["auroc_cut25"] == pytest.approx(1.0)
    assert res["auroc_macro"] == pytest.approx(1.0)
    assert res["f1_macro"] == pytest.approx(1.0)


# --- Walk-forward ---------------------------------------------------------------


def test_walk_forward_trains_only_on_earlier_meetings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each prediction sees a strict prefix of the record: no future meetings."""
    X, y = _cyclic_xy(40)
    calls: list[pd.Index] = []
    real_fit = decision.fit_ordered_logit

    def spy(X_train: pd.DataFrame, y_train: pd.Series, **kw):
        calls.append(X_train.index)
        return real_fit(X_train, y_train, **kw)

    monkeypatch.setattr(decision, "fit_ordered_logit", spy)
    wf = walk_forward(X, y, min_train=10, refit_every=5)
    assert [len(c) for c in calls] == [10, 15, 20, 25, 30, 35]
    for train_idx in calls:
        assert list(train_idx) == list(y.index[: len(train_idx)])
    assert list(wf.index) == list(y.index[10:])
    assert list(wf.columns) == [*CLASSES, "label", "n_train"]
    assert (wf["n_train"].to_numpy() == np.arange(10, 40)).all()
    assert np.allclose(wf[list(CLASSES)].sum(axis=1), 1.0)
    assert list(wf["label"].cat.categories) == list(CLASSES)


def test_walk_forward_refuses_inconsistent_arguments() -> None:
    """A non-positive min_train or refit_every raises."""
    X, y = _cyclic_xy(12)
    with pytest.raises(ValueError, match="min_train"):
        walk_forward(X, y, min_train=0)
    with pytest.raises(ValueError, match="refit_every"):
        walk_forward(X, y, min_train=4, refit_every=0)


def test_feature_blocks_partition_features() -> None:
    """HISTORY, RULE_GAPS and MACRO partition the FEATURES columns."""
    assert set(HISTORY) | set(RULE_GAPS) | set(MACRO) == set(FEATURES)
    assert not (set(HISTORY) & set(RULE_GAPS)) and not (set(HISTORY) & set(MACRO))
    assert set(SPECS["full"]) == set(FEATURES)
    assert SPECS["history"] == HISTORY


def test_market_block_outside_features_and_macro() -> None:
    """MARKET holds the two bill spreads and leaves MACRO and "full" alone."""
    assert MARKET == ("tbill3_spread", "tbill6_spread")
    assert not set(MARKET) & set(FEATURES)
    assert SPECS["market"] == MARKET
    assert SPECS["history+market"] == (*HISTORY, *MARKET)


def test_headline_structure_synthetic() -> None:
    """headline returns the history-only walk-forward and its evaluation."""
    dates = _meeting_dates(12)
    out = headline(_monthly_frame(_month_starts(dates)), _meeting_frame(dates, _SIZES),
                   min_train=6)
    assert out.columns == HISTORY
    assert list(out.walk_forward.columns) == [*CLASSES, "label", "n_train"]
    assert set(out.evaluation) == {
        "model", "expanding_prior", "prior_full_sample", "constant_hold"
    }
    assert np.allclose(out.walk_forward[list(CLASSES)].sum(axis=1), 1.0)


def test_headline_beats_the_expanding_prior() -> None:
    """On the real record the history-only headline beats the leak-free prior.

    The headline specification is the three-feature model of the committee's
    own recent behaviour: the ablation shows macro levels alone score worse
    than the class prior out-of-sample, so the parsimonious model is the
    defensible default. Refits happen every meeting (the default); with stale
    parameters the model falls behind the prior.
    """
    out = headline()
    assert len(out.walk_forward) == 171
    res = out.evaluation
    assert res["model"]["rps"] < res["expanding_prior"]["rps"]
    assert res["model"]["rps"] < res["constant_hold"]["rps"]
    assert res["model"]["kappa"] > 0.0


def test_ablation_structure_synthetic() -> None:
    """ablation returns one model row per spec and three reference rows."""
    dates = _meeting_dates(12)
    X = features(_monthly_frame(_month_starts(dates)), _meeting_frame(dates, _SIZES))
    y = labels(_meeting_frame(dates, _SIZES))
    table = ablation(X, y, min_train=6)
    assert list(table.columns[:7]) == ["spec", "kind", "k", "kappa", "rps", "log_loss", "accuracy"]
    assert {"auroc_macro", "auprc_macro", "f1_macro", "balanced_accuracy"} <= set(table.columns)
    models = table[table["kind"] == "model"]
    assert list(models["spec"]) == list(SPECS)
    assert list(models["k"]) == [len(SPECS[s]) for s in SPECS]
    refs = table[table["kind"] == "reference"]
    assert list(refs["spec"]) == ["expanding_prior", "prior_full_sample", "constant_hold"]
    assert np.isfinite(table["rps"]).all()


@pytest.mark.slow
def test_ablation_ordering_on_real_record() -> None:
    """The ablation finding: history-only beats the full spec and the prior.

    The claimed ordering is the one the 271-decision record supports
    out-of-sample: the three-feature history model has the best RPS, the full
    19-feature specification is worse, and macro levels alone lose to the
    leak-free expanding prior.
    """
    table = ablation(features(), labels(), specs=["history", "full", "macro"])
    rps = table.set_index("spec")["rps"]
    assert rps["history"] < rps["full"]
    assert rps["history"] < rps["expanding_prior"]
    assert rps["macro"] > rps["expanding_prior"]


def test_cut25_diagnosis_structure_synthetic() -> None:
    """The diagnosis dict has the documented keys on synthetic data."""
    dates = _meeting_dates(12)
    X = features(_monthly_frame(_month_starts(dates)), _meeting_frame(dates, _SIZES))
    y = labels(_meeting_frame(dates, _SIZES))
    out = cut25_diagnosis(X, y, min_train=6)
    assert {"cut25_dates", "cut50_dates", "predictions", "means", "cohens_d",
            "loo_accuracy", "recall"} <= set(out)
    assert len(out["cut25_dates"]) == 2  # -50 and -25 sizes in _SIZES
    assert len(out["cut50_dates"]) == 2
    assert 0.0 <= out["loo_accuracy"] <= 1.0
    assert set(out["recall"]) == {"cut25", "cut50+"}
    assert out["predictions"].index.isin(y.index).all()


@pytest.mark.slow
def test_cut25_diagnosis_real_record() -> None:
    """The cut25 blind spot: a macro signature, no usable separation.

    The insurance-cut hypothesis holds descriptively - cut25 meetings see
    positive payroll growth, calm financial conditions and few recessions
    while cut50+ meetings are intermeeting, recession-adjacent and stressed -
    but the classes overlap: leave-one-out nearest-centroid on the pooled
    cut25/cut50+ subset reaches only modest accuracy (0.5-0.75 against a 0.475
    majority base rate) across feature sets, and the headline model never gives
    cut25 the argmax at any evaluation-window cut25 meeting - 8 of 11 go to
    hold. The blind spot is cut25 versus hold at decision time, not cut25
    versus cut50+.
    """
    out = cut25_diagnosis()
    d = out["cohens_d"]
    assert d["intermeeting"] < -0.5
    assert d["payems_chg1"] > 0.5
    assert d["nfci"] < -0.5
    assert 0.5 <= out["loo_accuracy"] < 0.75
    assert 0.3 <= out["recall"]["cut25"] < 0.9
    preds = out["predictions"]
    assert (preds["predicted"] == "cut25").mean() < 0.2
    assert set(preds.columns) == {"predicted", "p_cut25"}
