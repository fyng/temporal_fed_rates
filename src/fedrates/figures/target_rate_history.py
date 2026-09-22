"""Production chart: thirty years of announced fed funds target decisions.

Run:
    uv run python -m fedrates.figures.target_rate_history
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from fedrates import dataset, fomc, register, save
from fedrates.theme import GREY_LABEL, MAIN, annotate_event, economist, period_shading

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"
_X_START = pd.Timestamp("1994-01-01")
_X_END = pd.Timestamp("2026-10-01")
_RANGE_ERA = pd.Timestamp("2008-12-16")

TITLE = "Thirty years of decisions"
SUBTITLE = "United States, fed funds target midpoint, %"
SOURCE = "Source: Federal Reserve; FRED; NBER"
FOOTNOTE = "Band shows the target range, in place since Dec 2008; shaded areas mark NBER recessions"

_TICKVALS = [f"{y}-01-01" for y in (1995, 2000, 2005, 2010, 2015, 2020, 2025)]
_TICKTEXT = ["1995", "2000", "05", "10", "15", "20", "25"]

# (decision date, label, ax, ay): the fewest marks that carry the message.
_EVENTS = (
    ("1994-02-04", "First announced change,<br>Feb 4th 1994", 95, 35),
    ("2008-12-16", "Cut to 0-0.25%", -40, -30),
    ("2015-12-16", "Liftoff", 12, -30),
    ("2020-03-15", "Back to zero", 8, -30),
    ("2023-07-26", "525bp, Mar 2022 - Jul 2023", 18, -30),
    ("2026-09-16", "3.75-4.00%", -45, -72),
)


def _mid_target(meetings: pd.DataFrame) -> pd.Series:
    """Target midpoint as a step series across the 2008 point-to-range splice.

    Args:
        meetings: Committed FOMC decision record.

    Returns:
        Midpoint indexed by decision date, extended to the axis end.
    """
    mid = meetings["target"].fillna((meetings["target_lower"] + meetings["target_upper"]) / 2.0)
    dates = pd.DatetimeIndex(list(meetings["date"]) + [_X_END])
    return pd.Series(list(mid) + [mid.iloc[-1]], index=dates)


def _band(meetings: pd.DataFrame) -> pd.DataFrame:
    """Target range bounds as a step series over the range era.

    Args:
        meetings: Committed FOMC decision record.

    Returns:
        Frame of date, target_lower, target_upper extended to the axis end.
    """
    post = meetings.loc[meetings["date"] >= _RANGE_ERA, ["date", "target_lower", "target_upper"]]
    tail = pd.DataFrame(
        {
            "date": [_X_END],
            "target_lower": [post["target_lower"].iloc[-1]],
            "target_upper": [post["target_upper"].iloc[-1]],
        }
    )
    return pd.concat([post, tail], ignore_index=True)


def _recessions() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """NBER spans inside the chart window, ends extended to full months.

    Returns:
        (start, end) pairs shaped for period_shading.
    """
    return [
        (s, e + pd.offsets.MonthBegin(1))
        for s, e in dataset.recession_spans()
        if e >= _X_START
    ]


def build_fig() -> go.Figure:
    """Build the target history chart.

    Returns:
        Decorated Economist figure.
    """
    register()
    meetings = fomc.load_meetings()
    mid = _mid_target(meetings)
    band = _band(meetings)
    fig = go.Figure()
    fig.add_scatter(
        x=mid.index,
        y=mid.to_numpy(),
        mode="lines",
        name="Target midpoint",
        line=dict(color=MAIN["BLUE"], width=2.2),
        line_shape="hv",
    )
    fig.add_scatter(
        x=band["date"],
        y=band["target_upper"],
        mode="lines",
        name="Target range",
        line=dict(color=GREY_LABEL, width=0.75),
        line_shape="hv",
        legendgroup="band",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=band["date"],
        y=band["target_lower"],
        mode="lines",
        name="Target range",
        line=dict(color=GREY_LABEL, width=0),
        fill="tonexty",
        fillcolor="rgba(164,189,201,0.25)",
        line_shape="hv",
        legendgroup="band",
        showlegend=False,
        hoverinfo="skip",
    )
    period_shading(fig, _recessions())
    for date, text, ax, ay in _EVENTS:
        annotate_event(fig, date, text, y=float(mid.asof(pd.Timestamp(date))), ax=ax, ay=ay)
    economist(
        fig,
        title=TITLE,
        subtitle=SUBTITLE,
        source=SOURCE,
        footnote=FOOTNOTE,
        size=SIZE,
        legend=True,
        zeroline=False,
    )
    fig.update_xaxes(range=[_X_START, _X_END], tickvals=_TICKVALS, ticktext=_TICKTEXT)
    fig.update_yaxes(range=[0, 7.15], dtick=1)
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("target_rate_history.png", "target_rate_history.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
