"""Tests for fedrates.models.level."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedrates import dataset, fomc, fred, sources
from fedrates.models import level

# --- Synthetic frames ---------------------------------------------------------


def _ar1(
    rng: np.random.Generator, rho: float, sd: float, n: int, index: pd.DatetimeIndex
) -> pd.Series:
    """Gaussian AR(1) with zero start."""
    x = np.empty(n)
    x[0] = 0.0
    for s in range(1, n):
        x[s] = rho * x[s - 1] + rng.normal(0.0, sd)
    return pd.Series(x, index=index)


def _simulate(
    rho: float = 0.85, a: float = 0.5, b: float = 1.0, *, pi_star: float = 2.0,
    n: int = 900, seed: int = 42,
) -> pd.DataFrame:
    """Frame generated exactly from the level model with known parameters.

    Inflation gap and activity gap are independent AR(1)s, r* drifts slowly,
    and u_gap carries the Okun sign (slack positive) so the fitted b should
    recover the b multiplying the output-convention gap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1950-01-01", periods=n, freq="MS", name="date")
    r_star = pd.Series(2.0 + 0.5 * np.sin(np.arange(n) / 80.0), index=idx)
    pi_gap = _ar1(rng, 0.9, 0.35, n, idx)
    gap = _ar1(rng, 0.9, 0.8, n, idx)
    e = rng.normal(0.0, 0.25, n)
    rate = np.empty(n)
    rate[0] = r_star.iloc[0] + pi_star + pi_gap.iloc[0]
    for s in range(1, n):
        target = (
            r_star.iloc[s] + (pi_star + pi_gap.iloc[s]) + a * pi_gap.iloc[s] + b * gap.iloc[s]
        )
        rate[s] = rho * rate[s - 1] + (1.0 - rho) * target + e[s]
    return pd.DataFrame(
        {
            "policy_rate": rate,
            "pce_core_yoy": pi_star + pi_gap,
            "u_gap": -gap,
            "r_star": r_star,
        },
        index=idx,
    )


def _hand_fit(*, n: int, start: pd.Timestamp, end: pd.Timestamp) -> level.LevelFit:
    """LevelFit with hand-set coefficients for exact predict checks."""
    return level.LevelFit(
        rho=0.85, rho_se=0.01, a=0.5, a_se=0.10, b=1.0, b_se=0.10,
        reduced=pd.Series({"c1": 0.85, "c2": 0.075, "c3": 0.15}),
        reduced_se=pd.Series({"c1": 0.01, "c2": 0.01, "c3": 0.01}),
        long_run_pi=1.5, long_run_gap=1.0, pi_star=2.0, gap="u_gap",
        r2=0.99, lags=6, start=start, end=end, n=n,
    )


# --- Estimator: recovery on synthetic data -------------------------------------


@pytest.mark.parametrize("seed", [42, 7])
def test_recovery_recovers_known_parameters(seed: int) -> None:
    truths = ({"rho": 0.85, "a": 0.5, "b": 1.0}, {"rho": 0.9, "a": 1.0, "b": 0.5})
    for truth in truths:
        frame = _simulate(seed=seed, **truth)
        f = level.fit(frame)
        assert f.rho == pytest.approx(truth["rho"], abs=0.03)
        assert f.a == pytest.approx(truth["a"], abs=0.30)
        assert f.b == pytest.approx(truth["b"], abs=0.15)
        # Estimates stay within four HAC standard errors of the truth.
        assert abs(f.rho - truth["rho"]) < 4.0 * f.rho_se
        assert abs(f.a - truth["a"]) < 4.0 * f.a_se
        assert abs(f.b - truth["b"]) < 4.0 * f.b_se
        # The reduced form is exact: c1 = rho, c2 = (1-rho)(1+a), c3 = (1-rho)b.
        assert f.reduced["c1"] == pytest.approx(f.rho)
        assert f.reduced["c2"] == pytest.approx(
            (1.0 - truth["rho"]) * (1.0 + truth["a"]), abs=0.03
        )
        assert f.reduced["c3"] == pytest.approx((1.0 - truth["rho"]) * truth["b"], abs=0.03)
        assert np.isfinite(f.reduced_se.to_numpy()).all()
        assert (f.reduced_se > 0.0).all()


def test_delta_se_positive_and_finite_on_synthetic() -> None:
    f = level.fit(_simulate())
    for se in (f.rho_se, f.a_se, f.b_se):
        assert np.isfinite(se) and se > 0.0


def test_pi_star_is_load_bearing() -> None:
    frame = _simulate()
    right = level.fit(frame, pi_star=2.0)
    wrong = level.fit(frame, pi_star=3.0)
    assert abs(wrong.a - 0.5) > abs(right.a - 0.5)


def test_fit_respects_sample_bounds() -> None:
    frame = _simulate(n=300)
    f_all = level.fit(frame)
    f_sub = level.fit(frame, start="1955-01-01", end="1965-12-31")
    assert f_sub.start >= pd.Timestamp("1955-01-01")
    assert f_sub.end <= pd.Timestamp("1965-12-31")
    assert f_sub.n < f_all.n
    assert f_sub.start > f_all.start and f_sub.end < f_all.end


def test_bad_gap_raises() -> None:
    with pytest.raises(ValueError, match="gap"):
        level.fit(_simulate(n=60), gap="bogus")


# --- predict -------------------------------------------------------------------


def test_predict_hand_computed() -> None:
    idx = pd.date_range("2022-01-01", periods=13, freq="MS", name="date")
    frame = pd.DataFrame(
        {"policy_rate": 4.0, "pce_core_yoy": 3.0, "u_gap": -1.0, "r_star": 2.0}, index=idx
    )
    pred = level.predict(_hand_fit(n=13, start=idx[0], end=idx[-1]), frame)
    assert pred.name == "policy_rate_fitted"
    assert pd.isna(pred.iloc[0])
    # rho·i_{t-1} + (1-rho)·(r* + pi + a·(pi - pi*) + b·(-u_gap)) = 0.85·4 + 0.15·6.5
    assert np.allclose(pred.iloc[1:], 0.85 * 4.0 + 0.15 * 6.5)


# --- Real frame (offline from the local cache) ----------------------------------


@pytest.fixture(scope="module")
def real_frame() -> pd.DataFrame:
    """Real monthly analysis frame, strictly offline from the on-disk cache."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fred, "is_stale", lambda path, max_age_days=1.0: False)
        mp.setattr(sources, "download", lambda url, path, **kw: path)
        return dataset.load_frame()


@pytest.fixture(scope="module")
def quarterly_frame() -> pd.DataFrame:
    """Real quarterly analysis frame, built strictly offline from the cache."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fred, "is_stale", lambda path, max_age_days=1.0: False)
        mp.setattr(sources, "download", lambda url, path, **kw: path)
        return dataset.build_frame("q")


def test_rho_high_on_real_frame(real_frame: pd.DataFrame) -> None:
    f = level.fit(real_frame)
    assert 0.0 < f.rho < 1.0
    assert f.rho > 0.8


def test_gap_coefficients_positive_on_real_frame(real_frame: pd.DataFrame) -> None:
    # u_gap enters negated (Okun), so b > 0 means tightening as the gap rises.
    f = level.fit(real_frame)
    assert f.a > 0.0
    assert f.b > 0.0


def test_hac_se_positive_finite_and_lags_honoured(real_frame: pd.DataFrame) -> None:
    f = level.fit(real_frame)
    for se in (f.rho_se, f.a_se, f.b_se):
        assert np.isfinite(se) and se > 0.0
    assert level.fit(real_frame, lags=12).lags == 12


def test_predict_contract_on_real_frame(real_frame: pd.DataFrame) -> None:
    f = level.fit(real_frame)
    pred = level.predict(f, real_frame)
    assert pred.index.equals(real_frame.index)
    assert pred.index.name == "date"
    assert pred.name == "policy_rate_fitted"
    assert pd.isna(pred.iloc[0])
    finite = pred.dropna()
    assert len(finite) > 0.9 * len(pred)
    actual = real_frame["policy_rate"].loc[finite.index].to_numpy()
    assert np.corrcoef(finite.to_numpy(), actual)[0, 1] > 0.95


def test_fit_by_chair_flags_and_warsh_nan(real_frame: pd.DataFrame) -> None:
    out = level.fit_by_chair(real_frame)
    assert list(out["chair"]) == [name for name, _, _ in fomc.CHAIRS]
    assert list(out.columns) == [
        "chair", "start", "end", "n", "rho", "rho_se", "identified", "a", "a_se", "b", "b_se",
        "r2",
    ]
    yellen = out.loc[out["chair"] == "Yellen"].iloc[0]
    assert yellen["start"] == pd.Timestamp("2014-02-01")
    assert yellen["end"] == pd.Timestamp("2018-02-04")
    # rho is reported and finite for every fitted tenure; long-run coefficients
    # only where identified — Greenspan alone (rho < 0.99 and both SEs < 2;
    # Yellen clears the rho bar but not the SE bar).
    fitted = out.iloc[:-1]
    assert fitted["rho"].notna().all() and (fitted["n"] >= 24).all()
    assert out.loc[out["identified"], "chair"].tolist() == ["Greenspan"]
    assert out.loc[out["identified"], ["a", "a_se", "b", "b_se"]].notna().all().all()
    unidentified = out.loc[~out["identified"] & out["rho"].notna()]
    assert unidentified[["a", "a_se", "b", "b_se"]].isna().all().all()
    warsh = out.iloc[-1]
    assert warsh["chair"] == "Warsh"
    assert pd.isna(warsh["end"])
    assert warsh["n"] < 24  # too few usable periods, and the row says so
    assert not warsh["identified"]
    assert warsh[["rho", "a", "b"]].isna().all()


def test_fit_headline_is_quarterly_post_1983(
    real_frame: pd.DataFrame, quarterly_frame: pd.DataFrame
) -> None:
    f = level.fit_headline(quarterly_frame)
    assert f.start == level.HEADLINE_START
    assert 0.0 < f.rho < 1.0 and f.rho > 0.8
    g = level.fit(quarterly_frame, start="1983-01-01")
    assert (f.rho, f.a, f.b, f.n) == (g.rho, g.a, g.b, g.n)
    with pytest.raises(ValueError, match="quarterly"):
        level.fit_headline(real_frame)


def test_reduced_form_exposed_on_real_frame(real_frame: pd.DataFrame) -> None:
    f = level.fit(real_frame)
    assert list(f.reduced.index) == ["c1", "c2", "c3"]
    assert f.reduced["c1"] == pytest.approx(f.rho)
    assert np.isfinite(f.reduced.to_numpy()).all()
    assert np.isfinite(f.reduced_se.to_numpy()).all() and (f.reduced_se > 0.0).all()
    # The recovered long-run coefficients are exact functions of the reduced form.
    d = 1.0 - f.rho
    assert f.a == pytest.approx(f.reduced["c2"] / d - 1.0)
    assert f.b == pytest.approx(f.reduced["c3"] / d)


def test_specification_grid_contract_and_identification(
    real_frame: pd.DataFrame, quarterly_frame: pd.DataFrame
) -> None:
    grid = level.specification_grid(real_frame, quarterly_frame)
    assert len(grid) == 6
    assert list(grid.columns) == [
        "freq", "start", "n", "rho", "rho_se", "a", "a_se", "t_a", "b", "b_se", "t_b",
        "long_run_pi", "r2",
    ]
    assert set(grid["freq"]) == {"monthly", "quarterly"}
    assert list(grid.loc[grid["freq"] == "quarterly", "start"]) == ["1961+", "1983+", "1994+"]
    assert (grid["n"] > 0).all()
    assert np.isfinite(grid[["rho", "a", "b"]].to_numpy()).all()
    # Found identification pattern: a never reaches 2 SE from zero in any cell;
    # b is positive throughout and >= 2 SE in the 1983+ cells.
    assert (grid["t_a"].abs() < 2.0).all()
    assert (grid["b"] > 0.0).all()
    post83 = grid.loc[grid["start"] == "1983+"]
    assert (post83["t_b"] >= 2.0).all()
    # The one soft cell, reported rather than papered over: monthly 1994+ has b
    # positive but only ~1.8 SE out.
    m94 = grid.loc[(grid["freq"] == "monthly") & (grid["start"] == "1994+")].iloc[0]
    assert 0.0 < m94["t_b"] < 2.0


def test_rolling_contract(real_frame: pd.DataFrame) -> None:
    full = level.fit(real_frame)
    out = level.rolling(real_frame)
    assert list(out.columns) == ["rho", "a", "b"]
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing
    assert len(out) == full.n - 120 + 1
    assert out.index[0] == full.start + pd.DateOffset(months=119)
    assert out.index[-1] == full.end
    assert np.isfinite(out.to_numpy()).all()


def test_oos_contract_and_random_walk_tie(real_frame: pd.DataFrame) -> None:
    full = level.fit(real_frame)
    out, summary = level.oos(real_frame)
    assert list(out.columns) == ["forecast", "actual", "model_error", "rw_error"]
    assert out.index.name == "date"
    assert len(out) == full.n - 120
    assert out.index[0] == full.start + pd.DateOffset(months=120)
    assert np.isfinite(out.to_numpy()).all()
    assert (out["actual"] - out["forecast"] - out["model_error"]).abs().max() < 1e-9
    prior = out.index[0] - pd.DateOffset(months=1)
    assert out["rw_error"].iloc[0] == pytest.approx(
        out["actual"].iloc[0] - real_frame.at[prior, "policy_rate"]
    )
    assert summary[["rmse_model", "rmse_rw"]].gt(0.0).all()
    assert summary["ratio"] == pytest.approx(summary["rmse_model"] / summary["rmse_rw"])
    # Found relationship: an expanding-window fit ties but does not beat the
    # one-month random walk on this sample (ratio ~1.015; variants 0.98-1.00).
    assert 0.95 <= summary["ratio"] <= 1.10


# --- Errors ---------------------------------------------------------------------


def test_window_out_of_range_raises(real_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="window"):
        level.rolling(real_frame, window=6)
    with pytest.raises(ValueError, match="window"):
        level.rolling(real_frame, window=10_000)


def test_min_train_too_big_raises(real_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="min_train"):
        level.oos(real_frame, min_train=100_000)
