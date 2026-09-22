"""Ordinal classifier for FOMC decisions: features, ordered logit, walk-forward.

The label is the decision on :data:`fomc.CLASSES`, an ordered five-class scale
from ``cut50+`` to ``hike50+``. Holds are two-thirds of the record, so accuracy
is uninformative; the metrics that matter are quadratic-weighted kappa and the
ranked probability score, both judged against the leak-free expanding prior in
:func:`evaluate`. Headline numbers come from :func:`walk_forward`, which is
strictly out-of-sample in date order - the record is a time series and shuffled
cross-validation would leak the future.

The headline specification is :data:`HISTORY`, the three-feature model of the
committee's own recent behaviour, per :func:`ablation`: macro levels alone
score worse than the class prior out-of-sample and the rule gaps alone are
indistinguishable from it, so the predictive content is what the committee
last did, not measured conditions. :func:`headline` runs it;
:data:`FEATURES` is the full design matrix retained for the ablation.

Every macro feature is joined through :func:`fomc.state_at_meetings`, the
leakage boundary: a meeting on date d sees only frame months before d.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

from .. import dataset, fomc

__all__ = [
    "CLASSES",
    "FEATURES",
    "HISTORY",
    "MACRO",
    "RULE_GAPS",
    "SPECS",
    "Headline",
    "ablation",
    "cut25_diagnosis",
    "evaluate",
    "features",
    "fit_ordered_logit",
    "headline",
    "labels",
    "predict_proba",
    "quadratic_weighted_kappa",
    "rps",
    "walk_forward",
]

CLASSES: tuple[str, ...] = fomc.CLASSES
_HOLD = CLASSES.index("hold")
_N_CLASSES = len(CLASSES)

FEATURES: dict[str, str] = {
    "pi_gap": "Core PCE inflation gap vs the 2% target",
    "u_gap": "Unemployment gap, u - u*",
    "y_gap": "Output gap, 100 log(y/y*)",
    "term_spread": "10-year minus 2-year Treasury yield",
    "nfci": "Chicago Fed financial conditions index",
    "vix": "VIX, monthly mean",
    "claims": "Initial jobless claims, monthly mean",
    "bei10": "10-year breakeven inflation, monthly mean",
    "mich": "Michigan inflation expectations, end of month",
    "r_star": "Neutral real rate used by the rules; constant when the frame "
        "ships r* = 2.0 and dropped by the fitter",
    "payems_chg1": "Change in nonfarm payrolls over one month, thousands",
    "rule_gap_taylor93": "Taylor (1993) prescription minus the prevailing target midpoint",
    "rule_gap_balanced": "Balanced-approach prescription minus the prevailing target midpoint",
    "rule_gap_shortfalls": "Shortfalls-rule prescription minus the prevailing target midpoint",
    "rule_gap_inertial": "Inertial-rule prescription minus the prevailing target midpoint",
    "rule_gap_first_diff": "First-difference rule prescription minus the prevailing "
        "target midpoint",
    "decision_lag1": "Ordinal code of the previous decision, 0-4 over CLASSES",
    "mtgs_since_move": "Prior meetings since the last non-hold decision, "
        "capped at the start of the record",
    "intermeeting": "1 for an intermeeting decision",
}

_PASSTHROUGH: tuple[str, ...] = (
    "pi_gap", "u_gap", "y_gap", "term_spread", "nfci", "vix", "claims", "bei10",
    "mich", "r_star",
)
_LEVEL_COLS: tuple[str, ...] = (
    *_PASSTHROUGH, "payems", "rule_taylor93", "rule_balanced", "rule_shortfalls",
    "rule_inertial", "rule_first_diff", "target_old", "target_upper", "target_lower",
)

# Feature blocks: the headline model uses HISTORY alone; see ablation().
HISTORY: tuple[str, ...] = ("decision_lag1", "mtgs_since_move", "intermeeting")
RULE_GAPS: tuple[str, ...] = (
    "rule_gap_taylor93", "rule_gap_balanced", "rule_gap_shortfalls",
    "rule_gap_inertial", "rule_gap_first_diff",
)
MACRO: tuple[str, ...] = tuple(
    c for c in FEATURES if c not in HISTORY and c not in RULE_GAPS
)
SPECS: dict[str, tuple[str, ...]] = {
    "history": HISTORY,
    "history+rule": (*HISTORY, *RULE_GAPS),
    "history+macro": (*HISTORY, *MACRO),
    "full": tuple(FEATURES),
    "rule": RULE_GAPS,
    "macro": MACRO,
}


# --- Features and labels ------------------------------------------------------


def features(frame: pd.DataFrame | None = None, meetings: pd.DataFrame | None = None,
             *, lag_months: int = 1) -> pd.DataFrame:
    """Design matrix for the decision model, one row per meeting.

    All macro columns are joined through :func:`fomc.state_at_meetings` at
    ``lag_months`` (payroll change needs one further month back), so no
    feature for meeting date d uses frame data from month d or later; see
    :data:`FEATURES` for column definitions. Rows come back in date order and
    carry NaN where the frame has no data yet; :func:`fit_ordered_logit`
    imputes them.

    Args:
        frame: Monthly analysis frame; ``dataset.load_frame()`` when None.
        meetings: Meetings frame; ``fomc.load_meetings()`` when None.
        lag_months: Publication lag enforced by the leakage boundary.

    Returns:
        Frame indexed by meeting date with the :data:`FEATURES` columns.
    """
    frame = dataset.load_frame() if frame is None else frame
    meetings = fomc.load_meetings() if meetings is None else meetings
    m = meetings.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(m["date"], name="date")
    state = fomc.state_at_meetings(frame, m, lag_months=lag_months, cols=list(_LEVEL_COLS))
    pay_prev = fomc.state_at_meetings(frame, m, lag_months=lag_months + 1, cols=["payems"])

    out = pd.DataFrame(index=dates)
    for col in _PASSTHROUGH:
        out[col] = state[col].to_numpy()
    out["payems_chg1"] = state["payems"].to_numpy() - pay_prev["payems"].to_numpy()
    mid_prev = _prev_target_midpoint(m, state)
    for key in ("taylor93", "balanced", "shortfalls", "inertial", "first_diff"):
        out[f"rule_gap_{key}"] = state[f"rule_{key}"].to_numpy() - mid_prev
    codes = fomc.label_actions(m).cat.codes.to_numpy()
    pos = np.arange(codes.size)
    last_move = np.maximum.accumulate(np.where(codes != _HOLD, pos, -1))
    prior_move = np.concatenate(([-1], last_move[:-1]))
    out["decision_lag1"] = pd.Series(codes.astype(float), index=dates).shift(1)
    out["mtgs_since_move"] = np.where(prior_move >= 0, pos - prior_move, pos).astype(float)
    out["intermeeting"] = m["intermeeting"].to_numpy(dtype=float)
    return out[list(FEATURES)]


def _prev_target_midpoint(m: pd.DataFrame, state: pd.DataFrame) -> np.ndarray:
    """Target midpoint in force entering each meeting, as an array.

    The midpoint set by the previous decision; the record's first meeting
    falls back to the frame's target columns for the lagged month.

    Args:
        m: Meetings sorted by date with target, target_lower, target_upper.
        state: Lagged frame state on the meeting-date index.

    Returns:
        Array of midpoints aligned with ``m``.
    """
    mid = m["target"].where(m["target"].notna(), (m["target_lower"] + m["target_upper"]) / 2.0)
    prev = np.array(mid.shift(1), dtype=float)
    if np.isnan(prev[0]):
        frame_mid = state["target_old"].where(
            state["target_old"].notna(), (state["target_upper"] + state["target_lower"]) / 2.0)
        prev[0] = float(frame_mid.iloc[0])
    return prev


def labels(meetings: pd.DataFrame | None = None) -> pd.Series:
    """Ordered decision labels over :data:`CLASSES`, indexed by meeting date.

    Args:
        meetings: Meetings frame; ``fomc.load_meetings()`` when None.

    Returns:
        Ordered categorical Series named ``label`` on the meeting-date index.
    """
    m = fomc.load_meetings() if meetings is None else meetings
    m = m.sort_values("date")
    out = fomc.label_actions(m)
    out.index = pd.DatetimeIndex(m["date"], name="date")
    return out


# --- Model --------------------------------------------------------------------


@dataclass
class OrderedLogit:
    """Fitted proportional-odds logit with its preprocessing state.

    Args:
        result: statsmodels OrderedModel results.
        medians: Fit-window medians used to fill NaN features.
        columns: Columns kept after dropping zero-variance features.
        mean: Fit-window means of the kept columns.
        scale: Fit-window standard deviations of the kept columns.
    """

    result: object
    medians: pd.Series
    columns: pd.Index
    mean: pd.Series
    scale: pd.Series


def fit_ordered_logit(X: pd.DataFrame, y: pd.Series, *, distr: str = "logit",
                      maxiter: int = 5000) -> OrderedLogit:
    """Fit a proportional-odds ordered logit of the label on the features.

    NaN features are filled with fit-window medians, zero-variance columns
    dropped and the rest standardised, so the coefficients read as effects
    per feature standard deviation. ``y`` is coerced onto the :data:`CLASSES`
    order. A bfgs fit that fails to converge is retried with Nelder-Mead.

    Args:
        X: Feature frame as produced by :func:`features`.
        y: Labels; ordered categorical, integer codes or class names.
        distr: Link distribution, "logit" or "probit".
        maxiter: Optimiser iteration cap.

    Returns:
        The fitted wrapper for :func:`predict_proba`.
    """
    y_cat = _as_categorical(y)
    medians = X.median(numeric_only=True)
    filled = X.fillna(medians)
    keep = filled.std() > 1e-12
    kept = filled.loc[:, keep]
    mean = kept.mean()
    scale = kept.std().replace(0.0, 1.0)
    model = OrderedModel(y_cat, (kept - mean) / scale, distr=distr)
    return OrderedLogit(
        result=_fit(model, maxiter), medians=medians, columns=kept.columns,
        mean=mean, scale=scale,
    )


def _fit(model: OrderedModel, maxiter: int) -> object:
    """Fit with bfgs, retrying with Nelder-Mead when bfgs does not converge.

    Args:
        model: The OrderedModel to fit.
        maxiter: Optimiser iteration cap.

    Returns:
        The statsmodels results object.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        warnings.simplefilter("ignore", OptimizeWarning)
        res = model.fit(method="bfgs", disp=False, maxiter=maxiter)
    if res.mle_retvals.get("converged", False):
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(method="nm", disp=False, maxiter=maxiter)


def predict_proba(model: OrderedLogit, X: pd.DataFrame) -> pd.DataFrame:
    """Class probabilities from a fitted ordered logit.

    Args:
        model: Wrapper from :func:`fit_ordered_logit`.
        X: Feature frame with the fit input's columns.

    Returns:
        Frame indexed like ``X`` with the :data:`CLASSES` columns; each row
        sums to 1.
    """
    scaled = (X.fillna(model.medians)[model.columns] - model.mean) / model.scale
    probs = np.asarray(
        model.result.model.predict(model.result.params, scaled.to_numpy(), which="prob")
    )
    cats = list(model.result.model.labels)
    if cats != list(CLASSES):
        aligned = np.zeros((probs.shape[0], _N_CLASSES))
        aligned[:, [CLASSES.index(c) for c in cats]] = probs
        probs = aligned
    return pd.DataFrame(probs, index=X.index, columns=list(CLASSES))


# --- Metrics ------------------------------------------------------------------


def quadratic_weighted_kappa(y_true: pd.Series | np.ndarray,
                             y_pred: pd.Series | np.ndarray) -> float:
    """Quadratic-weighted Cohen's kappa over the ordered classes.

    Args:
        y_true: Labels; ordered categorical, integer codes or class names.
        y_pred: Predicted labels in the same form.

    Returns:
        Kappa in [-1, 1]; 1 for perfect agreement, 0 for chance, and 0 for
        any constant predictor.
    """
    return float(cohen_kappa_score(
        _codes(y_true), _codes(y_pred), weights="quadratic", labels=range(_N_CLASSES),
    ))


def rps(y_true: pd.Series | np.ndarray, proba: pd.DataFrame | np.ndarray) -> float:
    """Ranked probability score over the ordered classes; lower is better.

    RPS = mean over meetings of (1 / (K - 1)) sum_k (F_k - O_k)^2, where F
    and O are the predicted and realised cumulative distributions and
    K = 5. It scores the whole predictive distribution and is 0 only for
    perfect predictions.

    Args:
        y_true: Labels; ordered categorical, integer codes or class names.
        proba: n x 5 class probabilities in :data:`CLASSES` order.

    Returns:
        Mean RPS over the meetings given.
    """
    codes = _codes(y_true)
    p = _proba_array(proba)
    cumulative_p = np.cumsum(p, axis=1)[:, :-1]
    hit = np.zeros_like(p)
    hit[np.arange(codes.size), codes] = 1.0
    cumulative_o = np.cumsum(hit, axis=1)[:, :-1]
    return float(((cumulative_p - cumulative_o) ** 2).sum(axis=1).mean() / (_N_CLASSES - 1))


def evaluate(y_true: pd.Series | np.ndarray,
             proba: pd.DataFrame | np.ndarray) -> dict[str, dict[str, float]]:
    """Score predicted probabilities against three baselines.

    Args:
        y_true: Realised labels, in date order; ordered categorical, integer
            codes or class names.
        proba: n x 5 predicted probabilities in :data:`CLASSES` order.

    Returns:
        Dict mapping ``model`` / ``expanding_prior`` / ``prior_full_sample`` /
        ``constant_hold``, each to ``kappa``, ``rps``, ``log_loss`` and
        ``accuracy``. ``expanding_prior`` is the leak-free headline
        comparison: the empirical class distribution of ``y_true`` up to each
        meeting, uniform before any history. ``prior_full_sample`` peeks at
        the realised class frequencies of the whole window and is a secondary
        reference only; ``constant_hold`` is the degenerate all-hold
        distribution and scores infinite log loss wherever a non-hold label
        occurs.
    """
    codes = _codes(y_true)
    onehot = np.zeros((codes.size, _N_CLASSES))
    onehot[:, _HOLD] = 1.0
    freq = np.bincount(codes, minlength=_N_CLASSES) / codes.size
    out: dict[str, dict[str, float]] = {}
    for name, p in (
        ("model", _proba_array(proba)),
        ("expanding_prior", _expanding_prior(codes)),
        ("prior_full_sample", np.tile(freq, (codes.size, 1))),
        ("constant_hold", onehot),
    ):
        pred = p.argmax(axis=1)
        out[name] = {
            "kappa": quadratic_weighted_kappa(codes, pred),
            "rps": rps(codes, p),
            "log_loss": _log_loss(codes, p),
            "accuracy": float((pred == codes).mean()),
        }
    return out


def _expanding_prior(codes: np.ndarray) -> np.ndarray:
    """Empirical class prior from earlier labels only; uniform before any.

    Args:
        codes: Integer label codes in date order.

    Returns:
        n x 5 array; row i is the class distribution of rows 0..i-1, or
        uniform for row 0.
    """
    counts = np.zeros(_N_CLASSES)
    out = np.empty((codes.size, _N_CLASSES))
    for i, c in enumerate(codes):
        out[i] = counts / i if i else np.full(_N_CLASSES, 1.0 / _N_CLASSES)
        counts[c] += 1
    return out


def _log_loss(y_true: pd.Series | np.ndarray, proba: pd.DataFrame | np.ndarray) -> float:
    """Mean negative log probability of the realised class.

    Args:
        y_true: Labels; ordered categorical, integer codes or class names.
        proba: n x 5 class probabilities in :data:`CLASSES` order.

    Returns:
        Mean log loss; infinite where a realised class gets probability 0.
    """
    codes = _codes(y_true)
    p = _proba_array(proba)
    with np.errstate(divide="ignore"):
        return float((-np.log(p[np.arange(codes.size), codes])).mean())


def _codes(y: pd.Series | np.ndarray) -> np.ndarray:
    """Labels as integer codes on the CLASSES scale.

    Args:
        y: Labels; ordered categorical, integer codes or class names.

    Returns:
        Array of codes 0-4.
    """
    return np.asarray(_as_categorical(y).cat.codes)


def _proba_array(proba: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Probability frame or array as an n x 5 array in CLASSES order.

    Args:
        proba: Class probabilities, by column name or position.

    Returns:
        Float array of probabilities.

    Raises:
        KeyError: If a DataFrame lacks one of the CLASSES columns.
    """
    if isinstance(proba, pd.DataFrame):
        proba = proba[list(CLASSES)]
    return np.asarray(proba, dtype=float)


def _as_categorical(y: pd.Series | np.ndarray) -> pd.Series:
    """Coerce labels onto the CLASSES scale as an ordered categorical Series.

    Args:
        y: Labels; ordered categorical, integer codes or class names.

    Returns:
        Ordered categorical Series with the CLASSES categories.

    Raises:
        ValueError: If integer codes fall outside 0-4.
    """
    if isinstance(y, pd.Series) and isinstance(y.dtype, pd.CategoricalDtype):
        return y.astype(pd.CategoricalDtype(CLASSES, ordered=True))
    index = y.index if isinstance(y, pd.Series) else None
    arr = np.asarray(y)
    if pd.api.types.is_integer_dtype(arr):
        if arr.size and (arr.min() < 0 or arr.max() >= _N_CLASSES):
            raise ValueError(f"codes must lie in 0-{_N_CLASSES - 1}")
        cat = pd.Categorical.from_codes(arr, list(CLASSES), ordered=True)
    else:
        cat = pd.Categorical(arr, categories=list(CLASSES), ordered=True)
    return pd.Series(cat, index=index, name="label")


# --- Out-of-sample evaluation ---------------------------------------------------


def walk_forward(X: pd.DataFrame, y: pd.Series, *, min_train: int = 100,
                 refit_every: int = 1) -> pd.DataFrame:
    """Strictly out-of-sample expanding-window predictions, one meeting at a time.

    Meeting i is predicted by a model fitted on meetings 0..i-1 only; the
    model is refitted every ``refit_every`` meetings and reused in between.
    Windows follow date order and are never shuffled: the record is a time
    series and shuffling would leak the future.

    Args:
        X: Feature frame indexed by meeting date.
        y: Labels; a Series is reindexed onto ``X`` when needed.
        min_train: Meetings in the first training window.
        refit_every: Meetings between refits.

    Returns:
        Frame indexed by meeting date with the five class-probability
        columns, the realised ``label`` and ``n_train``, the meetings used
        for that prediction.

    Raises:
        ValueError: If the arguments are inconsistent.
    """
    if min_train < 1 or refit_every < 1:
        raise ValueError("min_train and refit_every must be at least 1")
    y_cat = _as_categorical(y)
    if isinstance(y, pd.Series) and not y_cat.index.equals(X.index):
        y_cat = y_cat.reindex(X.index)
    codes = np.asarray(y_cat.cat.codes)
    if codes.size != len(X):
        raise ValueError("X and y lengths differ")
    probas: list[np.ndarray] = []
    n_train: list[int] = []
    model: OrderedLogit | None = None
    for i in range(min_train, len(X)):
        if model is None or (i - min_train) % refit_every == 0:
            model = fit_ordered_logit(X.iloc[:i], y_cat.iloc[:i])
        probas.append(predict_proba(model, X.iloc[i : i + 1]).to_numpy()[0])
        n_train.append(i)
    out = pd.DataFrame(probas, index=X.index[min_train:], columns=list(CLASSES))
    out["label"] = y_cat.iloc[min_train:]
    out["n_train"] = n_train
    return out


# --- Headline, ablation and diagnosis -------------------------------------------


@dataclass
class Headline:
    """The parsimonious history-only model and its out-of-sample evaluation.

    Args:
        columns: The :data:`HISTORY` features used.
        walk_forward: Per-meeting out-of-sample probabilities and labels.
        evaluation: :func:`evaluate` dict for the walk-forward output.
    """

    columns: tuple[str, ...]
    walk_forward: pd.DataFrame
    evaluation: dict[str, dict[str, float]]


def headline(frame: pd.DataFrame | None = None, meetings: pd.DataFrame | None = None,
             *, lag_months: int = 1, min_train: int = 100,
             refit_every: int = 1) -> Headline:
    """Run the headline history-only walk-forward and evaluate it.

    Args:
        frame: Monthly analysis frame; ``dataset.load_frame()`` when None.
        meetings: Meetings frame; ``fomc.load_meetings()`` when None.
        lag_months: Publication lag enforced by the leakage boundary.
        min_train: Meetings in the first training window.
        refit_every: Meetings between refits.

    Returns:
        The :class:`Headline` for the :data:`HISTORY` specification.
    """
    X = features(frame, meetings, lag_months=lag_months)
    y = labels(meetings)
    wf = walk_forward(X[list(HISTORY)], y, min_train=min_train, refit_every=refit_every)
    return Headline(
        columns=HISTORY, walk_forward=wf,
        evaluation=evaluate(wf["label"], wf[list(CLASSES)]),
    )


def ablation(X: pd.DataFrame, y: pd.Series, *, min_train: int = 100, refit_every: int = 1,
             specs: Sequence[str] | None = None) -> pd.DataFrame:
    """Out-of-sample walk-forward for each feature block, with the baselines.

    Args:
        X: Feature frame as produced by :func:`features`.
        y: Labels aligned with ``X``.
        min_train: Meetings in the first training window.
        refit_every: Meetings between refits.
        specs: Names from :data:`SPECS` to run; all of them when None.

    Returns:
        Frame with columns ``spec, kind, k, kappa, rps, log_loss, accuracy``:
        one row per specification (kind ``model``) then one per baseline
        (kind ``reference``), the expanding prior first.

    Raises:
        KeyError: If a spec name is not in :data:`SPECS`.
    """
    rows: list[dict[str, object]] = []
    wf = None
    for name in (SPECS if specs is None else specs):
        cols = list(SPECS[name])
        wf = walk_forward(X[cols], y, min_train=min_train, refit_every=refit_every)
        res = evaluate(wf["label"], wf[list(CLASSES)])
        rows.append({"spec": name, "kind": "model", "k": len(cols), **res["model"]})
    res = evaluate(wf["label"], wf[list(CLASSES)])
    for base in ("expanding_prior", "prior_full_sample", "constant_hold"):
        rows.append({"spec": base, "kind": "reference", "k": 0, **res[base]})
    return pd.DataFrame(rows)[["spec", "kind", "k", "kappa", "rps", "log_loss", "accuracy"]]


def cut25_diagnosis(X: pd.DataFrame | None = None, y: pd.Series | None = None,
                    frame: pd.DataFrame | None = None,
                    meetings: pd.DataFrame | None = None,
                    *, min_train: int = 100) -> dict[str, object]:
    """Test whether cut25 decisions carry a signature distinct from cut50+.

    Compares the two classes on every feature plus the NBER recession flag
    (joined at the leakage boundary), then checks bivariate separability with
    a leave-one-out nearest-centroid classifier on standardised features.
    Predictions come from the headline history-only walk-forward. Diagnosis
    only: no model is retuned to the small cut25 class.

    Args:
        X: Feature frame; ``features()`` when None.
        y: Labels; ``labels()`` when None.
        frame: Monthly analysis frame for the recession flag;
            ``dataset.load_frame()`` when None.
        meetings: Meetings frame; ``fomc.load_meetings()`` when None.
        min_train: Meetings in the first training window.

    Returns:
        Dict with ``cut25_dates`` and ``cut50_dates`` (record), ``predictions``
        (eval-window cut25 meetings: predicted class and P(cut25) from the
        headline model), ``means`` (feature by class), ``cohens_d``
        (cut25 minus cut50+, pooled standard deviations), ``loo_accuracy``
        and ``recall`` (per class) for the two-class separation check.
    """
    X = features() if X is None else X
    y = labels(meetings) if y is None else y
    frame = dataset.load_frame() if frame is None else frame
    meetings = fomc.load_meetings() if meetings is None else meetings
    m = meetings.sort_values("date").reset_index(drop=True)
    recession = fomc.state_at_meetings(frame, m, lag_months=1, cols=["recession"])["recession"]
    recession.index = pd.DatetimeIndex(m["date"], name="date")

    wf = walk_forward(X[list(HISTORY)], y, min_train=min_train)
    proba = wf[list(CLASSES)]
    predicted = pd.Series(
        np.asarray(CLASSES, dtype=object)[proba.to_numpy().argmax(axis=1)], index=wf.index
    )

    cut25 = y.index[y == "cut25"]
    cut50 = y.index[y == "cut50+"]
    state = X.join(recession).fillna(X.median())
    var = state.std() > 1e-12
    state = state.loc[:, var]
    pooled = np.sqrt(
        ((len(cut25) - 1) * state.loc[cut25].var() + (len(cut50) - 1) * state.loc[cut50].var())
        / (len(cut25) + len(cut50) - 2)
    )
    d = (state.loc[cut25].mean() - state.loc[cut50].mean()) / pooled

    scaled = StandardScaler().fit_transform(state.loc[cut25.union(cut50)])
    truth = (y.loc[cut25.union(cut50)] == "cut50+").to_numpy()
    loo = np.empty(scaled.shape[0], dtype=int)
    for i in range(scaled.shape[0]):
        train = np.arange(scaled.shape[0]) != i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo[i] = NearestCentroid().fit(scaled[train], truth[train]).predict(scaled[i:i+1])[0]

    dates = wf.index
    preds = pd.DataFrame(
        {"predicted": predicted.loc[cut25.intersection(dates)],
         "p_cut25": proba["cut25"].loc[cut25.intersection(dates)]}
    )
    return {
        "cut25_dates": cut25,
        "cut50_dates": cut50,
        "predictions": preds,
        "means": pd.DataFrame(
            {"cut25": state.loc[cut25].mean(), "cut50+": state.loc[cut50].mean()}
        ),
        "cohens_d": d,
        "loo_accuracy": float((loo == truth).mean()),
        "recall": {
            "cut25": float((loo[~truth] == 0).mean()),
            "cut50+": float((loo[truth] == 1).mean()),
        },
    }
