"""Inertial Fed reaction function estimated on the analysis frame.

Fits the partial-adjustment rule

    i_t = rho·i_{t-1} + (1 − rho)·(r*_t + pi_t + a·(pi_t − pi*) + b·g_t) + e_t

where i is ``policy_rate`` (effective rate with the Wu-Xia shadow rate at the
ELB), pi is ``pce_core_yoy`` and g the activity gap. Estimation is OLS on the
restricted reduced form — (i_t − r*_t − pi*) on [i_{t-1} − r*_t − pi*,
pi_t − pi*, g_t] with no constant — which imposes the model's unit long-run
response to r* and its implied level, and makes ``pi_star`` load-bearing.
(rho, a, b) are recovered by the delta method with HAC (Newey-West) standard
errors.

Headline specification: quarterly, post-1983, via :func:`fit_headline`. The
full monthly sample spans the 1983 reaction-function break, and monthly
aggregation pushes rho toward 1 because the committee holds at two-thirds of
its meetings; quarterly post-1983 is the literature convention (Taylor 1993;
Clarida, Galí and Gertler 2000). Monthly stays reachable as a robustness case
through :func:`specification_grid`.

Identification: a and b are recovered as c2/(1−rho) and c3/(1−rho), so their
precision degrades as rho approaches 1. Across the specification grid, a is
never more than ~1.6 standard errors from zero — it is not separately
identified at any sample or frequency checked — while b is positive in every
cell and at least 2 standard errors out in all but the monthly 1994+ cell
(1.8). Quote the well-estimated reduced-form coefficients
(``LevelFit.reduced``) wherever the division is unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import rules
from ..fomc import CHAIRS

__all__ = [
    "HEADLINE_START",
    "LevelFit",
    "fit",
    "fit_headline",
    "predict",
    "fit_by_chair",
    "rolling",
    "oos",
    "specification_grid",
]

PI_COL = "pce_core_yoy"
RATE_COL = "policy_rate"
RSTAR_COL = "r_star"

# Headline sample: the post-Volcker quarterly specification.
HEADLINE_START = pd.Timestamp("1983-01-01")

# Trailing-window contract: between one year of periods and the usable sample.
_MIN_WINDOW = 12
# Chair tenures with fewer usable periods get NaN coefficients, not estimates.
_MIN_CHAIR_N = 24
# A long-run coefficient is quotable only when the (1 − rho) division holds up:
# rho below 0.99 and both long-run HAC standard errors under 2.0 points.
_IDENT_RHO_MAX = 0.99
_IDENT_SE_MAX = 2.0
_LONG_RUN_COLS = ("a", "a_se", "b", "b_se")

_GAP_TO_KIND = {"u_gap": "unemployment", "y_gap": "output"}

_GRID_STARTS: tuple[tuple[str, str], ...] = (
    ("1961-01-01", "1961+"),
    ("1983-01-01", "1983+"),
    ("1994-01-01", "1994+"),
)


@dataclass(frozen=True)
class LevelFit:
    """Estimated inertial reaction function.

    a and b are recovered as c2/(1−rho) and c3/(1−rho), so their precision
    degrades as rho approaches 1; when the division is unstable, report
    ``reduced`` instead.

    Attributes:
        rho: Smoothing parameter on the lagged rate.
        rho_se: HAC standard error of rho.
        a: Long-run inflation-gap coefficient.
        a_se: HAC standard error of a.
        b: Long-run activity-gap coefficient.
        b_se: HAC standard error of b.
        reduced: Reduced-form coefficients c1, c2, c3 multiplying
            (i_{t-1} − r* − pi*), (pi − pi*) and g; c1 equals rho.
        reduced_se: HAC standard errors of the reduced-form coefficients.
        long_run_pi: Implied long-run response to inflation, 1 + a.
        long_run_gap: Implied long-run response to the gap, b.
        pi_star: Inflation target the fit assumed.
        gap: Frame column supplying the activity gap.
        r2: Centered R-squared of the restricted regression.
        lags: HAC (Newey-West) lag length used.
        start: First usable observation.
        end: Last usable observation.
        n: Usable observations.
    """

    rho: float
    rho_se: float
    a: float
    a_se: float
    b: float
    b_se: float
    reduced: pd.Series
    reduced_se: pd.Series
    long_run_pi: float
    long_run_gap: float
    pi_star: float
    gap: str
    r2: float
    lags: int
    start: pd.Timestamp
    end: pd.Timestamp
    n: int


# --- Design -------------------------------------------------------------------


def _signed_gap(frame: pd.DataFrame, gap: str) -> pd.Series:
    """Signed activity gap: −u_gap or +y_gap, per the ``rules`` convention.

    Args:
        frame: Analysis frame.
        gap: "u_gap" or "y_gap".

    Returns:
        The gap series with the sign convention applied.

    Raises:
        ValueError: If ``gap`` is not a known gap column.
    """
    kind = _GAP_TO_KIND.get(gap)
    if kind is None:
        raise ValueError(f"gap must be one of {sorted(_GAP_TO_KIND)}, got {gap!r}")
    return rules._gap_input(frame, {"u_gap": gap, "y_gap": gap}, kind)


def _usable(
    frame: pd.DataFrame,
    gap: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    pi_star: float,
) -> pd.DataFrame:
    """Restricted reduced-form design, rows with missing inputs dropped.

    The lag is taken on the full frame before clipping, so the first row of a
    window conditions on the observed rate from the period before it.

    Args:
        frame: Analysis frame.
        gap: Frame column supplying the activity gap.
        start: Inclusive lower bound; None keeps the frame start.
        end: Inclusive upper bound; None keeps the frame end.
        pi_star: Inflation target in percent.

    Returns:
        Frame with columns ``y, lag, pi, gap, r_star`` on the usable index.
    """
    df = pd.DataFrame(
        {
            "y": frame[RATE_COL] - frame[RSTAR_COL] - pi_star,
            "lag": frame[RATE_COL].shift(1) - frame[RSTAR_COL] - pi_star,
            "pi": frame[PI_COL] - pi_star,
            "gap": _signed_gap(frame, gap),
            "r_star": frame[RSTAR_COL],
        }
    ).loc[slice(start, end)]
    return df.dropna()


def _nw_lags(n: int) -> int:
    """Newey-West (1994) plug-in lag count, floor(4·(n/100)^(2/9)).

    Args:
        n: Number of observations.

    Returns:
        Lag count, at least 1.
    """
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _estimate(df: pd.DataFrame, *, pi_star: float, gap: str, lags: int | None) -> LevelFit:
    """Run the restricted OLS on a usable design frame and recover (rho, a, b).

    Args:
        df: Usable design frame from ``_usable``.
        pi_star: Inflation target the design was built with.
        gap: Frame column supplying the activity gap.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        The fit.

    Raises:
        ValueError: If fewer than 3 usable observations remain.
    """
    if len(df) < 3:
        raise ValueError(f"need at least 3 usable observations, got {len(df)}")
    if lags is None:
        lags = _nw_lags(len(df))
    res = sm.OLS(df["y"], df[["lag", "pi", "gap"]]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(lags), "use_correction": True}
    )
    cov = res.cov_params()
    reduced = pd.Series(
        {
            "c1": float(res.params["lag"]),
            "c2": float(res.params["pi"]),
            "c3": float(res.params["gap"]),
        }
    )
    reduced_se = pd.Series(
        {
            "c1": float(np.sqrt(cov.loc["lag", "lag"])),
            "c2": float(np.sqrt(cov.loc["pi", "pi"])),
            "c3": float(np.sqrt(cov.loc["gap", "gap"])),
        }
    )
    rho = float(reduced["c1"])
    d = 1.0 - rho
    a = float(reduced["c2"]) / d - 1.0
    b = float(reduced["c3"]) / d
    # Delta method: a and b are functions of (c1, c2) and (c1, c3).
    g_a = np.array([reduced["c2"] / d**2, 1.0 / d])
    g_b = np.array([reduced["c3"] / d**2, 1.0 / d])
    a_se = float(np.sqrt(g_a @ cov.loc[["lag", "pi"], ["lag", "pi"]].to_numpy() @ g_a))
    b_se = float(np.sqrt(g_b @ cov.loc[["lag", "gap"], ["lag", "gap"]].to_numpy() @ g_b))
    tss = float(((df["y"] - df["y"].mean()) ** 2).sum())
    return LevelFit(
        rho=rho,
        rho_se=float(reduced_se["c1"]),
        a=a,
        a_se=a_se,
        b=b,
        b_se=b_se,
        reduced=reduced,
        reduced_se=reduced_se,
        long_run_pi=1.0 + a,
        long_run_gap=b,
        pi_star=pi_star,
        gap=gap,
        r2=1.0 - float(res.ssr) / tss,
        lags=int(lags),
        start=df.index[0],
        end=df.index[-1],
        n=len(df),
    )


# --- Public API -----------------------------------------------------------------


def fit(
    frame: pd.DataFrame,
    *,
    gap: str = "u_gap",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    pi_star: float = 2.0,
    lags: int | None = None,
) -> LevelFit:
    """Estimate the inertial reaction function on the analysis frame.

    Frequency-agnostic: pass the monthly or quarterly analysis frame. The
    headline specification — quarterly, post-1983 — is :func:`fit_headline`.

    Args:
        frame: Analysis frame with ``policy_rate``, ``pce_core_yoy``,
            ``r_star`` and the gap column.
        gap: Frame column supplying the activity gap; ``u_gap`` enters
            negated (Okun), ``y_gap`` as-is.
        start: Inclusive sample start; None keeps the frame start.
        end: Inclusive sample end; None keeps the frame end.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        The fit, carrying (rho, a, b) with HAC standard errors, the
        reduced-form coefficients, the implied long-run coefficients,
        R-squared, the lag count and sample bounds.
    """
    df = _usable(frame, gap, start, end, pi_star=pi_star)
    return _estimate(df, pi_star=pi_star, gap=gap, lags=lags)


def fit_headline(
    frame_q: pd.DataFrame, *, gap: str = "u_gap", pi_star: float = 2.0, lags: int | None = None
) -> LevelFit:
    """Headline fit: the quarterly frame from :data:`HEADLINE_START`.

    Args:
        frame_q: Quarterly analysis frame (``dataset.build_frame("q")``).
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        The fit on ``frame_q`` from 1983-01-01.

    Raises:
        ValueError: If the frame looks monthly rather than quarterly.
    """
    freq = pd.infer_freq(frame_q.index)
    if freq is not None and freq.upper().startswith("M"):
        raise ValueError(f"fit_headline expects the quarterly frame, got a {freq} index")
    return fit(frame_q, gap=gap, start=HEADLINE_START, pi_star=pi_star, lags=lags)


def _rule_base(
    frame: pd.DataFrame,
    r_star: float | pd.Series,
    *,
    gap: str,
    pi_star: float,
    a: float,
    b: float,
    cols: dict[str, str] | None = None,
) -> pd.Series:
    """Unconstrained rule target r* + pi + a·(pi − pi*) + b·g for ``rules.inertial``.

    Args:
        frame: Analysis frame.
        r_star: Neutral real rate, broadcast onto the frame index.
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        a: Inflation-gap coefficient.
        b: Activity-gap coefficient.
        cols: Unused; present to match the base-rule protocol.

    Returns:
        Prescribed policy rate on the frame index.
    """
    pi = frame[PI_COL]
    return r_star + pi + a * (pi - pi_star) + b * _signed_gap(frame, gap)


def predict(fit: LevelFit, frame: pd.DataFrame) -> pd.Series:
    """Fitted policy-rate path: rho·i_{t-1} + (1 − rho)·(r* + pi + a·Δpi + b·g).

    Conditions on the observed lagged rate, as the regression does. NaN where
    the lag or any input is missing.

    Args:
        fit: Fit supplying (rho, a, b, pi_star, gap).
        frame: Analysis frame on the index to predict over.

    Returns:
        Fitted path named ``policy_rate_fitted`` on the frame index.
    """
    return rules.inertial(
        frame,
        r_star=frame[RSTAR_COL],
        rho=fit.rho,
        rate_col=RATE_COL,
        base=_rule_base,
        gap=fit.gap,
        pi_star=fit.pi_star,
        a=fit.a,
        b=fit.b,
    ).rename("policy_rate_fitted")


def _identified(f: LevelFit) -> bool:
    """Whether the long-run coefficients survive the (1 − rho) division.

    Args:
        f: The fit.

    Returns:
        True when rho is below 0.99 and both long-run HAC standard errors sit
        under 2.0 percentage points.
    """
    return bool(
        f.rho < _IDENT_RHO_MAX
        and np.isfinite(f.a_se)
        and f.a_se < _IDENT_SE_MAX
        and np.isfinite(f.b_se)
        and f.b_se < _IDENT_SE_MAX
    )


def fit_by_chair(
    frame: pd.DataFrame, *, gap: str = "u_gap", pi_star: float = 2.0, lags: int | None = None
) -> pd.DataFrame:
    """Run :func:`fit` once per chair tenure in ``fomc.CHAIRS``.

    Tenures with fewer than 24 usable periods return NaN coefficients and
    their usable count, rather than an estimate. Long-run coefficients are
    returned only where :func:`_identified` holds; elsewhere they are NaN
    under ``identified = False`` and only rho — well estimated regardless —
    is reported. Every post-Greenspan tenure fails the flag, so its a/b would
    be division noise, not findings.

    Args:
        frame: Analysis frame.
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        Frame with columns ``chair, start, end, n, rho, rho_se, identified,
        a, a_se, b, b_se, r2``; ``end`` is NaT for the open-ended tenure.
    """
    rows: list[dict] = []
    for name, start, end in CHAIRS:
        tenure_end = pd.Timestamp(end) if end else pd.NaT
        n = len(_usable(frame, gap, start, end or None, pi_star=pi_star))
        row: dict = {"chair": name, "start": pd.Timestamp(start), "end": tenure_end, "n": n}
        if n < _MIN_CHAIR_N:
            row["identified"] = False
            row.update({k: float("nan") for k in ("rho", "rho_se", "r2", *_LONG_RUN_COLS)})
        else:
            f = fit(frame, gap=gap, start=start, end=end or None, pi_star=pi_star, lags=lags)
            row["rho"] = f.rho
            row["rho_se"] = f.rho_se
            row["identified"] = _identified(f)
            if row["identified"]:
                row.update({"a": f.a, "a_se": f.a_se, "b": f.b, "b_se": f.b_se, "r2": f.r2})
            else:
                row.update({k: float("nan") for k in _LONG_RUN_COLS})
                row["r2"] = f.r2
        rows.append(row)
    return pd.DataFrame(rows)


def specification_grid(
    frame_m: pd.DataFrame, frame_q: pd.DataFrame, *, gap: str = "u_gap", pi_star: float = 2.0,
    lags: int | None = None,
) -> pd.DataFrame:
    """Fit every {monthly, quarterly} x {1961+, 1983+, 1994+} cell.

    This grid is the robustness evidence behind the headline specification
    and the identification finding: a is never 2 standard errors from zero in
    any cell, b is positive throughout and at least 2 standard errors out in
    the 1983+ cells.

    Args:
        frame_m: Monthly analysis frame.
        frame_q: Quarterly analysis frame.
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        One row per cell with columns ``freq, start, n, rho, rho_se, a, a_se,
        t_a, b, b_se, t_b, long_run_pi, r2``.
    """
    rows: list[dict] = []
    for freq, frame in (("monthly", frame_m), ("quarterly", frame_q)):
        for start, label in _GRID_STARTS:
            f = fit(frame, gap=gap, start=start, pi_star=pi_star, lags=lags)
            rows.append(
                {
                    "freq": freq,
                    "start": label,
                    "n": f.n,
                    "rho": f.rho,
                    "rho_se": f.rho_se,
                    "a": f.a,
                    "a_se": f.a_se,
                    "t_a": f.a / f.a_se,
                    "b": f.b,
                    "b_se": f.b_se,
                    "t_b": f.b / f.b_se,
                    "long_run_pi": f.long_run_pi,
                    "r2": f.r2,
                }
            )
    return pd.DataFrame(rows)


def rolling(
    frame: pd.DataFrame, window: int = 120, *, gap: str = "u_gap", pi_star: float = 2.0,
    lags: int | None = None,
) -> pd.DataFrame:
    """Trailing-window estimates of (rho, a, b) over the usable sample.

    Windows are contiguous stretches of usable observations, so the first
    estimate sits at the ``window``-th usable period.

    Args:
        frame: Analysis frame.
        window: Window length in usable periods.
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        Frame with columns ``rho, a, b`` indexed by the window end dates.

    Raises:
        ValueError: If ``window`` is under a year of periods or exceeds the
            usable sample.
    """
    df = _usable(frame, gap, pi_star=pi_star)
    if not _MIN_WINDOW <= window <= len(df):
        raise ValueError(f"window must be between {_MIN_WINDOW} and {len(df)}, got {window}")
    vals = [
        _estimate(df.iloc[i - window + 1 : i + 1], pi_star=pi_star, gap=gap, lags=lags)
        for i in range(window - 1, len(df))
    ]
    out = pd.DataFrame(
        [(f.rho, f.a, f.b) for f in vals],
        index=pd.DatetimeIndex(df.index[window - 1:], name="date"),
        columns=["rho", "a", "b"],
    )
    return out


def oos(
    frame: pd.DataFrame, *, min_train: int = 120, gap: str = "u_gap", pi_star: float = 2.0,
    lags: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Expanding-window one-step-ahead forecasts against a random-walk benchmark.

    Each forecast date t is predicted from a fit on every usable observation
    before it; the regressors at t (lagged rate, inflation, gap, r*) are taken
    as known. The benchmark is i_t = i_{t-1}.

    Args:
        frame: Analysis frame.
        min_train: Usable observations required before the first forecast.
        gap: Frame column supplying the activity gap.
        pi_star: Inflation target in percent.
        lags: HAC lag count; None uses the Newey-West plug-in.

    Returns:
        (frame, summary). The frame is indexed by forecast date with columns
        ``forecast, actual, model_error, rw_error`` (errors as actual minus
        prediction). The summary series carries ``rmse_model, rmse_rw`` and
        ``ratio`` = rmse_model / rmse_rw.
    """
    df = _usable(frame, gap, pi_star=pi_star)
    if min_train >= len(df):
        raise ValueError(f"min_train {min_train} leaves no forecast dates in {len(df)} rows")
    rows: list[dict] = []
    for t in range(min_train, len(df)):
        f = _estimate(df.iloc[:t], pi_star=pi_star, gap=gap, lags=lags)
        r = df.iloc[t]
        i_prev = r["lag"] + r["r_star"] + pi_star
        target = r["r_star"] + (r["pi"] + pi_star) + f.a * r["pi"] + f.b * r["gap"]
        pred = f.rho * i_prev + (1.0 - f.rho) * target
        date = df.index[t]
        actual = float(frame.at[date, RATE_COL])
        rows.append(
            {
                "forecast": pred,
                "actual": actual,
                "model_error": actual - pred,
                "rw_error": actual - i_prev,
            }
        )
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(df.index[min_train:], name="date"))
    rmse_model = float(np.sqrt((out["model_error"] ** 2).mean()))
    rmse_rw = float(np.sqrt((out["rw_error"] ** 2).mean()))
    summary = pd.Series(
        {"rmse_model": rmse_model, "rmse_rw": rmse_rw, "ratio": rmse_model / rmse_rw},
        name="oos",
    )
    return out, summary
