"""Production chart: the model's predicted decision at the eleven 25bp cuts.

Run:
    uv run python -m fedrates.figures.cut25_misses
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fedrates import register, save
from fedrates.models import decision
from fedrates.theme import MAIN, PALETTE, PT_TO_PX, TYPE_SCALE, economist

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
CACHE = Path(__file__).resolve().parents[3] / "data" / "processed" / "cut25_walk_forward.csv"
SIZE = "col3"

TITLE = "Insurance claims"
SUBTITLE = "Fed 25bp rate cuts, by the model's predicted decision"
SUBTITLE_2 = "Dec 2005-Sep 2026"
SOURCE = "Sources: Federal Reserve; FRED; our analysis"
FOOTNOTE = (
    "Ordered logit refit before each meeting; the model never gave "
    "a 25bp cut more than a 29% chance"
)

# Predicted classes in display order: dominant error first, the empty
# small-cut column last. Keys are decision.CLASSES names.
_CLASSES_DISPLAY = {"cut50+": "Cut, 50bp or more", "cut25": "Cut, 25bp", "hold": "Hold"}
_DISPLAY_ORDER = ("Hold", "Cut, 50bp or more", "Cut, 25bp")
_N_CUT25 = 11
_Y_MAX = 8.9
_LABEL_OFF = 0.18  # label offset above a bar top, in y data units
_GAP = 15 * PT_TO_PX  # the empty band between subtitle and plot, from theme


def _walk_forward() -> pd.DataFrame:
    """Headline walk-forward output, computed once and cached to data.

    Returns:
        Frame of class probabilities, realised labels and training-window
        sizes indexed by meeting date.
    """
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["date"], index_col="date")
    X = decision.features()
    y = decision.labels()
    wf = decision.walk_forward(X[list(decision.HISTORY)], y, min_train=100, refit_every=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    wf.to_csv(CACHE)
    return wf


def _predicted_counts(wf: pd.DataFrame) -> pd.Series:
    """Eval-window 25bp cuts counted by the model's predicted class.

    Args:
        wf: Walk-forward output from the headline history-only model.

    Returns:
        Counts indexed by display name in display order.

    Raises:
        ValueError: If the window does not hold exactly eleven 25bp cuts.
    """
    proba = wf[list(decision.CLASSES)]
    predicted = pd.Series(
        np.asarray(decision.CLASSES, dtype=object)[proba.to_numpy().argmax(axis=1)],
        index=wf.index,
    )
    misses = predicted.loc[wf.index[wf["label"] == "cut25"]]
    if len(misses) != _N_CUT25:
        raise ValueError(f"expected {_N_CUT25} 25bp cuts in the window, found {len(misses)}")
    counts = misses.value_counts().rename(_CLASSES_DISPLAY)
    return counts.reindex(_DISPLAY_ORDER, fill_value=0).astype(int)


def _font(role: str) -> dict:
    """Font dict for a type-scale role in house colours.

    Args:
        role: Key of TYPE_SCALE.

    Returns:
        Dict accepted by any plotly font attribute.
    """
    return dict(
        family=TYPE_SCALE[role]["family"],
        weight=TYPE_SCALE[role]["weight"],
        size=round(TYPE_SCALE[role]["size"] * PT_TO_PX, 1),
        color=PALETTE["TEXT"],
    )


def build_fig() -> go.Figure:
    """Build the 25bp-cut misses chart.

    Returns:
        Decorated Economist figure.
    """
    register()
    counts = _predicted_counts(_walk_forward())
    fig = go.Figure(
        go.Bar(
            x=list(counts.index),
            y=counts.to_list(),
            marker_color=MAIN["BLUE"],
            width=0.5,
            hoverinfo="skip",
        )
    )
    # Value labels as annotations, offset in data units so the label clears
    # the gridline at the full bar.
    for cat, v in counts.items():
        fig.add_annotation(
            x=cat,
            y=v + _LABEL_OFF,
            xref="x",
            yref="y",
            text=str(v),
            showarrow=False,
            yanchor="bottom",
            font=_font("annotation"),
        )
    economist(
        fig,
        title=TITLE,
        subtitle=SUBTITLE,
        source=SOURCE,
        footnote=FOOTNOTE,
        size=SIZE,
        legend=False,
        zeroline=False,
    )
    # The period rides as the subtitle's second line, in the empty band the
    # theme leaves between the subtitle and the plot.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        xanchor="left",
        yanchor="middle",
        yshift=_GAP / 2,
        text=SUBTITLE_2,
        showarrow=False,
        font=_font("period"),
    )
    fig.update_yaxes(range=[0, _Y_MAX], dtick=2)
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("cut25_misses.png", "cut25_misses.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
