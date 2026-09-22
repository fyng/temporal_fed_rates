"""Production chart: rolling inflation response of the Fed's rate rule.

Run:
    uv run python -m fedrates.figures.rolling_inflation_response
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fedrates import dataset, register, save
from fedrates.models import level
from fedrates.theme import BLACK, FONT_BODY, MAIN, TEXT, economist, period_shading

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"
_X_START = pd.Timestamp("1970-12-01")
_X_END = pd.Timestamp("2026-10-01")
_BREAK = pd.Timestamp("1983-01-01")
_UNSTABLE = pd.Timestamp("2000-01-01")
_Y_LO, _Y_HI = -3.0, 4.0

TITLE = "The Volcker break"
SUBTITLE = "Fed response to inflation, 10-year rolling estimate*"
SOURCE = "Sources: Federal Reserve; FRED; our analysis"
FOOTNOTE = (
    "*Long-run rise in the fed funds rate per point of inflation, beyond one-for-one; "
    "circles mark windows off the scale, which reach ±560"
)

_TICKVALS = [f"{y}-01-01" for y in (1975, 1985, 1995, 2005, 2015, 2025)]
_TICKTEXT = ["1975", "1985", "95", "05", "15", "25"]

# Medians run the full span they summarise; the 1983-99 median stops where the
# zero-rate era begins, because pooling 2000+ would pull it off the line.
_MEDIAN_SPANS = (
    (_X_START, _BREAK),
    (_BREAK, _UNSTABLE),
)
_ANNOT_FONT = dict(family=FONT_BODY, size=12.5, color=TEXT)
_PERIOD_FONT = dict(family=FONT_BODY, size=11.6, color=TEXT, weight=300)


def _medians(a: pd.Series) -> tuple[float, float]:
    """Medians of the inflation response before and after the 1983 break.

    Args:
        a: Rolling long-run inflation response indexed by window end date.

    Returns:
        (pre-1983, 1983-99) medians of ``a``; the second stops at 2000, whose
        windows are unstable enough to pool the median off the stable line.
    """
    pre = a.loc[a.index < _BREAK].median()
    mid = a.loc[(a.index >= _BREAK) & (a.index < _UNSTABLE)].median()
    return float(pre), float(mid)


def _signed(value: float) -> str:
    """Format a coefficient with an explicit sign and a true minus.

    Args:
        value: Coefficient.

    Returns:
        Text such as "−0.6" or "+0.8".
    """
    return f"{value:+.1f}".replace("-", "−")


def _period_label(fig: go.Figure, x: pd.Timestamp, text: str) -> None:
    """Label a shaded span at its top in the theme's period style.

    Args:
        fig: Figure with a shaded span on the x axis.
        x: Anchor date for the centred uppercase label.
        text: Label text, rendered uppercase.
    """
    fig.add_annotation(
        xref="x",
        yref="paper",
        x=x,
        y=1,
        yanchor="top",
        yshift=-3,
        text=text.upper(),
        showarrow=False,
        font=_PERIOD_FONT,
    )


def build_fig() -> go.Figure:
    """Build the rolling inflation response chart.

    Returns:
        Decorated Economist figure.
    """
    register()
    a = level.rolling(dataset.load_frame(), 120)["a"]
    pre, mid = _medians(a)
    in_range = a.where(a.between(_Y_LO, _Y_HI))
    off_scale = a.loc[~a.between(_Y_LO, _Y_HI)]
    fig = go.Figure()
    fig.add_scatter(
        x=a.index,
        y=in_range.to_numpy(),
        mode="lines",
        name="Window estimate",
        line=dict(color=MAIN["BLUE"], width=1.8),
    )
    fig.add_scatter(
        x=off_scale.index,
        y=np.where(off_scale.to_numpy() < 0, _Y_LO, _Y_HI),
        mode="markers",
        name="Off-scale window",
        marker=dict(symbol="circle-open", size=4, color=BLACK, line=dict(width=0.9)),
        cliponaxis=False,
    )
    period_shading(fig, [(_X_START, _BREAK), (_UNSTABLE, _X_END)])
    _period_label(fig, pd.Timestamp("1976-07-01"), "Before Volcker")
    _period_label(fig, pd.Timestamp("2018-01-01"), "Unstable")
    for (x0, x1), med in zip(_MEDIAN_SPANS, (pre, mid), strict=True):
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=med,
            y1=med,
            layer="above",
            line=dict(color=BLACK, width=0.9, dash="dash"),
        )
    fig.add_annotation(
        xref="x",
        yref="y",
        x=pd.Timestamp("1982-02-01"),
        y=pre,
        yshift=-7,
        xanchor="center",
        yanchor="top",
        text=f"Median {_signed(pre)}",
        showarrow=False,
        font=_ANNOT_FONT,
    )
    fig.add_annotation(
        xref="x",
        yref="y",
        x=pd.Timestamp("1998-10-01"),
        y=mid,
        yshift=7,
        xanchor="center",
        yanchor="bottom",
        text=f"Median {_signed(mid)}",
        showarrow=False,
        font=_ANNOT_FONT,
    )
    economist(
        fig,
        title=TITLE,
        subtitle=SUBTITLE,
        source=SOURCE,
        footnote=FOOTNOTE,
        size=SIZE,
        legend=False,
        zeroline="auto",
    )
    fig.update_xaxes(range=[_X_START, _X_END], tickvals=_TICKVALS, ticktext=_TICKTEXT)
    fig.update_yaxes(range=[_Y_LO, _Y_HI], dtick=1)
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("rolling_inflation_response.png", "rolling_inflation_response.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
