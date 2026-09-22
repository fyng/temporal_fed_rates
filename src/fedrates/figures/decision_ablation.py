"""Production chart: decision-model ablation, forecast error by feature set.

Run:
    uv run python -m fedrates.figures.decision_ablation

The walk-forward refits before every meeting and takes minutes per
specification; per-meeting probabilities are cached to data/processed/ (never
committed), so metrics recompute without refitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedrates import register, save
from fedrates.models.decision import CLASSES, SPECS, evaluate, features, labels, walk_forward
from fedrates.theme import (
    ECON_RED,
    GREY_LABEL,
    MAIN,
    PT_TO_PX,
    SIZES,
    TEXT,
    TYPE_SCALE,
    _font,
    _text_width,
    economist,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
CACHE = Path(__file__).resolve().parents[3] / "data" / "processed" / "decision_ablation.csv"
SIZE = "col3"

TITLE = "Past performance"
SUBTITLE = "Scores of Fed decision models, by inputs used, Dec 2005-Sep 2026"
SUBTITLE_2 = "Higher is better, except probability score; dashed line = baseline of past shares"
SOURCE = "Sources: Federal Reserve; FRED; our analysis"
FOOTNOTE = (
    "*Ranked probability score. Areas and F1 are one class against the rest, "
    "averaged over five decisions; models refit before each meeting"
)

# (spec, label) top to bottom: feature counts climb 3 -> 19, baselines last.
_ROWS: tuple[tuple[str, str], ...] = (
    ("history", "History only"),
    ("rule", "Rule gaps only"),
    ("history+rule", "History + rule gaps"),
    ("macro", "Economy only"),
    ("history+macro", "History + economy"),
    ("full", "Everything"),
    ("constant_hold", "Always hold"),
)
_FOCUS = "history"
_PRIOR = "expanding_prior"

# (metric, panel title, axis range, tick values) left to right.
_PANELS: tuple[tuple[str, str, tuple[float, float], tuple[float, ...]], ...] = (
    ("rps", "Probability score*", (0, 0.124), (0, 0.05, 0.1)),
    ("auroc_macro", "ROC area", (0, 1.02), (0, 0.5, 1)),
    ("auprc_macro", "Precision-recall area", (0, 1.02), (0, 0.5, 1)),
    ("f1_macro", "F1 score", (0, 1.02), (0, 0.5, 1)),
)

# Confirmed against the current record; the draft's prior row (0.0835) is
# stale. The build stops rather than plot off these.
_EXPECTED: dict[str, tuple[int, float]] = {
    "history": (3, 0.0523),
    "rule": (5, 0.0823),
    "history+rule": (8, 0.0580),
    "macro": (11, 0.1090),
    "history+macro": (14, 0.0726),
    "full": (19, 0.0793),
    "expanding_prior": (0, 0.0845),
    "constant_hold": (0, 0.0921),
}
_WF = CACHE.parent / "decision_wf_{spec}.csv"


def _walk_forward(spec: str, X: pd.DataFrame | None = None,
                  y: pd.Series | None = None) -> pd.DataFrame:
    """Per-meeting walk-forward probabilities for one specification, cached.

    Args:
        spec: Name from :data:`SPECS`.
        X: Feature frame; ``features()`` when None and not cached.
        y: Labels; ``labels()`` when None and not cached.

    Returns:
        Frame of class probabilities and realised ``label``, by meeting date.
    """
    path = Path(str(_WF).format(spec=spec.replace("+", "_")))
    if path.exists():
        return pd.read_csv(path, index_col="date", parse_dates=["date"])
    X = features() if X is None else X
    y = labels() if y is None else y
    wf = walk_forward(X[list(SPECS[spec])], y)
    wf.to_csv(path)
    return wf


def _ablation() -> pd.DataFrame:
    """Ablation table rebuilt from the cached walk-forward probabilities.

    Returns:
        Frame with spec, kind, k and every :func:`evaluate` metric per row:
        one per specification, then the three baselines.
    """
    X = y = None
    if not all(Path(str(_WF).format(spec=s.replace("+", "_"))).exists() for s in SPECS):
        X, y = features(), labels()
    rows = []
    for spec in SPECS:
        wf = _walk_forward(spec, X, y)
        res = evaluate(wf["label"], wf[list(CLASSES)])
        rows.append({"spec": spec, "kind": "model", "k": len(SPECS[spec]), **res["model"]})
    for base in ("expanding_prior", "prior_full_sample", "constant_hold"):
        rows.append({"spec": base, "kind": "reference", "k": 0, **res[base]})
    table = pd.DataFrame(rows)
    table.to_csv(CACHE, index=False)
    return table


def _check_expected(table: pd.DataFrame) -> None:
    """Stop when the computed ablation departs from the confirmed numbers.

    Args:
        table: Ablation frame from :func:`_ablation`.

    Raises:
        ValueError: If a feature count or score differs beyond rounding.
    """
    got = table.set_index("spec")
    bad = []
    for spec, (k, rps_pub) in _EXPECTED.items():
        k_got, rps_got = int(got.loc[spec, "k"]), float(got.loc[spec, "rps"])
        if k_got != k or abs(rps_got - rps_pub) > 5e-4:
            bad.append(f"{spec}: k {k_got} vs {k}, rps {rps_got:.4f} vs {rps_pub:.4f}")
    if bad:
        raise ValueError("ablation differs from the confirmed numbers:\n" + "\n".join(bad))


def _reanchor_tag(fig: go.Figure, size: str) -> None:
    """Re-anchor the red brand tag after margin changes.

    economist() fixes the tag's paper coordinates from the margins at call
    time; widening them afterwards leaves it adrift.

    Args:
        fig: Decorated figure whose tag is re-anchored in place.
        size: Named preset from SIZES the figure was built at.
    """
    w, h = SIZES[size]["px"]
    m = fig.layout.margin
    pw, ph = w - m.l - m.r, h - m.t - m.b
    tag_w, tag_h = 15 * PT_TO_PX, 5 * PT_TO_PX
    x0 = -m.l / pw
    y_top = (h - m.b) / ph
    for shape in fig.layout.shapes:
        if shape.fillcolor == ECON_RED:
            shape.x0 = x0
            shape.x1 = x0 + tag_w / pw
            shape.y0 = y_top - tag_h / ph
            shape.y1 = y_top


def build_fig() -> go.Figure:
    """Build the four-panel ablation chart.

    Returns:
        Decorated Economist figure.
    """
    register()
    table = _ablation()
    _check_expected(table)
    row = table.set_index("spec")
    specs = [spec for spec, _ in _ROWS]
    counts = [int(row.loc[spec, "k"]) for spec in specs]
    ticktext = [
        f"{name} ({k})" if k else name for (_, name), k in zip(_ROWS, counts, strict=True)
    ]
    colours = [MAIN["BLUE"] if spec == _FOCUS else GREY_LABEL for spec in specs]
    n = len(specs)
    fig = make_subplots(rows=1, cols=len(_PANELS), shared_yaxes=True, horizontal_spacing=0.035)
    for col, (metric, _, _, _) in enumerate(_PANELS, start=1):
        vals = [float(row.loc[spec, metric]) for spec in specs]
        # Plotly draws the first category at the bottom: reverse only the y
        # positions, so every value lands on its own label, top-down.
        fig.add_trace(
            go.Bar(
                x=vals,
                y=list(range(n))[::-1],
                orientation="h",
                marker=dict(color=colours),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        prior = float(row.loc[_PRIOR, metric])
        fig.add_shape(
            type="line",
            xref=f"x{col if col > 1 else ''}",
            yref="paper",
            x0=prior,
            x1=prior,
            y0=0,
            y1=1,
            line=dict(color=TEXT, width=0.72, dash="dash"),
            layer="above",
        )
    # Guard: every plotted bar carries its own specification's cached value.
    for trace, (metric, _, _, _) in zip(fig.data, _PANELS, strict=True):
        for pos, val in zip(trace.y, trace.x, strict=True):
            assert val == float(row.loc[specs[n - 1 - int(pos)], metric])
    economist(
        fig,
        title=TITLE,
        subtitle=SUBTITLE,
        source=SOURCE,
        footnote=FOOTNOTE,
        size=SIZE,
        legend=False,
        zeroline=False,
        yside="left",
    )
    # Bands of their own for the footnote (below the source), the second
    # subtitle line and the panel titles.
    w, _h = SIZES[SIZE]["px"]
    band = 10 * PT_TO_PX
    fig.update_layout(
        bargap=0.35,
        margin=dict(
            l=max(_text_width(t, "tick") for t in ticktext) + 14.0,
            r=5 * PT_TO_PX,
            t=fig.layout.margin.t + band,
            b=fig.layout.margin.b + band,
        ),
    )
    for ann in fig.layout.annotations:
        if ann.text == FOOTNOTE:
            ann.yshift -= band
        elif ann.text in (TITLE, SUBTITLE):
            ann.yshift += band
    pad = 5 * PT_TO_PX
    left = (pad - fig.layout.margin.l) / (w - fig.layout.margin.l - fig.layout.margin.r)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=left,
        y=1,
        xanchor="left",
        yanchor="bottom",
        yshift=(15 - TYPE_SCALE["subtitle"]["leading"]) * PT_TO_PX - 2.0 + band,
        text=SUBTITLE_2,
        showarrow=False,
        align="left",
        font=_font("sub-subtitle"),
    )
    for col, (_, name, _, _) in enumerate(_PANELS, start=1):
        fig.add_annotation(
            xref=f"x{col if col > 1 else ''} domain",
            yref="paper",
            x=0,
            y=1,
            xanchor="left",
            yanchor="bottom",
            yshift=3,
            text=name,
            showarrow=False,
            align="left",
            font=_font("annotation"),
        )
    # The house anchors the left furniture to the block edge, not the plot
    # area, which now sits a label column in from that edge.
    for ann in fig.layout.annotations:
        if ann.xref == "paper" and ann.x == 0:
            ann.x = left
    _reanchor_tag(fig, SIZE)
    for col, (_, _, rng, ticks) in enumerate(_PANELS, start=1):
        fig.update_xaxes(
            range=list(rng),
            tickvals=list(ticks),
            ticktext=[f"{t:g}" for t in ticks],
            row=1,
            col=col,
        )
    fig.update_yaxes(
        range=[-0.5, n - 0.5],
        tickvals=list(range(n)),
        ticktext=list(reversed(ticktext)),
        ticklabelstandoff=6,
        showgrid=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(showgrid=False, showticklabels=False, range=[-0.5, n - 0.5])
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    print(_ablation().to_string(index=False))
    for name in ("decision_ablation.png", "decision_ablation.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
