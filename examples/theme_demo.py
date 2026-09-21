"""Generate the Economist-style example figures into figures/examples/.

Run:
    uv run python examples/theme_demo.py

All data is synthetic, shaped like the real series (fed funds target path,
CPI, policy rates across economies). No network or API keys needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from fedrates import economist, register, save
from fedrates.theme import (
    MAIN,
    annotate_event,
    highlight,
    mpl_furniture,
    mpl_style,
    period_shading,
    signed,
)

OUT = Path(__file__).resolve().parents[1] / "figures" / "examples"

_YEARS = list(range(1994, 2026))
_YEAR_DATES = [f"{y}-01-01" for y in _YEARS]
_AXIS_END = "2025-10-01"

# Effective federal funds rate, monthly inflections, 1994-2025 (synthetic path).
_FED_FUNDS = [
    ("1994-01-03", 3.00), ("1994-11-15", 5.50), ("1995-02-01", 6.00),
    ("1996-01-31", 5.25), ("1997-03-25", 5.50),
    ("1998-10-15", 4.75), ("1999-06-30", 5.00), ("1999-11-16", 5.50),
    ("2000-05-16", 6.50), ("2001-01-03", 5.75), ("2001-03-20", 5.00),
    ("2001-05-15", 4.00), ("2001-09-17", 3.00), ("2001-12-11", 1.75),
    ("2002-11-06", 1.25), ("2003-06-25", 1.00), ("2004-06-30", 1.25),
    ("2004-12-14", 2.25), ("2005-06-30", 3.25), ("2005-12-13", 4.25),
    ("2006-06-29", 5.25), ("2007-09-18", 4.75), ("2007-12-11", 4.25),
    ("2008-03-18", 2.25), ("2008-10-08", 1.50), ("2008-12-16", 0.15),
    ("2015-12-16", 0.35), ("2017-03-15", 0.90), ("2018-03-21", 1.65),
    ("2019-08-01", 2.10), ("2019-10-30", 1.55), ("2020-03-16", 0.05),
    ("2022-03-17", 0.33), ("2022-09-22", 3.08), ("2022-12-15", 4.33),
    ("2023-05-04", 5.08), ("2023-07-27", 5.33), ("2024-09-19", 4.83),
    ("2024-12-19", 4.33), ("2025-09-18", 4.08),
]

_CPI = [2.6, 2.8, 3.0, 2.3, 1.6, 2.2, 3.4, 2.8, 1.6, 2.3, 2.7, 3.4, 3.2, 2.8, 3.8,
        -0.4, 1.6, 3.2, 2.1, 1.5, 1.6, 0.1, 1.3, 2.1, 2.4, 1.8, 1.2, 4.7, 8.0,
        4.1, 2.9, 3.0]

# Policy rates at end of year, %. Euro area begins in 1999.
_POLICY = {
    "United States": [5.5, 5.5, 5.25, 5.5, 4.75, 5.5, 6.5, 1.75, 1.25, 1.0, 2.25, 4.25, 5.25,
                      4.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.65, 1.4,
                      2.4, 1.55, 0.05, 0.05, 4.33, 5.33, 4.33, 4.08],
    "Euro area": [None, None, None, None, None, 3.0, 4.75, 3.25, 2.75, 2.0, 2.0, 2.25, 3.5,
                  4.0, 2.5, 1.0, 1.0, 1.0, 0.75, 0.25, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 2.0, 4.0, 3.15, 2.0],
    "Britain": [5.5, 6.25, 6.0, 7.25, 6.25, 5.5, 6.0, 4.0, 4.0, 4.0, 4.75, 4.5, 5.0, 5.5,
                2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.75, 0.75, 0.1,
                0.25, 3.5, 5.25, 4.75, 4.0],
    "Japan": [1.75, 0.5, 0.5, 0.5, 0.25, 0.15, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.4,
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
              -0.1, -0.1, -0.1, 0.25, 0.5],
}

# Yearly net change in the fed funds target, basis points.
_RATE_CHANGES = [
    (2015, 20), (2016, 30), (2017, 75), (2018, 100), (2019, -85), (2020, -150),
    (2022, 428), (2023, 100), (2024, -100), (2025, -25),
]


def _year_ticks(years: list[int], every: int = 5) -> tuple[list[str], list[str]]:
    """Build Economist-style year ticks: full label first and at century marks.

    Args:
        years: All years present on the axis.
        every: Spacing between ticks.

    Returns:
        (tickvals, ticktext) for a date axis.
    """
    sel = [y for y in years if (y - years[0]) % every == 0]
    text = [str(y) if i == 0 or y % 100 == 0 else f"{y % 100:02d}" for i, y in enumerate(sel)]
    return [f"{y}-01-01" for y in sel], text


def _apply_year_ticks(fig: go.Figure, every: int = 5) -> None:
    """Set abbreviated year ticks and axis range on a figure's first date axis.

    The range starts a year before the first tick; plotly hides tick labels
    that sit exactly on the range edge.

    Args:
        fig: Figure whose x axis spans 1994-2025.
        every: Spacing between ticks.
    """
    vals, text = _year_ticks([y for y in _YEARS if y >= 1995], every)
    fig.update_xaxes(range=["1994-01-01", _AXIS_END], tickvals=vals, ticktext=text)


def fig_single_line() -> go.Figure:
    """CPI line on the col2 preset.

    Returns:
        Styled single-line figure.
    """
    fig = go.Figure(go.Scatter(x=_YEAR_DATES, y=_CPI, line=dict(color=MAIN["BLUE"])))
    economist(
        fig,
        title="Inflation is back near target",
        subtitle="United States, consumer prices, % change on a year earlier",
        source="Source: FRED",
        legend=False,
    )
    _apply_year_ticks(fig)
    return fig


def fig_multi_line(size: str = "col2") -> go.Figure:
    """Four economies' policy rates.

    Args:
        size: Named size preset.

    Returns:
        Styled multi-line figure.
    """
    fig = go.Figure()
    for name, values in _POLICY.items():
        fig.add_trace(go.Scatter(x=_YEAR_DATES, y=values, name=name))
    economist(
        fig,
        title="Rates converge, slowly",
        subtitle="Central bank policy rate, %, end of year",
        source="Source: FRED; The Economist",
        size=size,
    )
    _apply_year_ticks(fig)
    return fig


def fig_highlight() -> go.Figure:
    """Six bond-yield series with one highlighted.

    Returns:
        Styled highlight figure.
    """
    rng = np.random.default_rng(7)
    starts = {"US": 6.6, "Germany": 6.9, "Japan": 4.6, "Britain": 8.0, "Canada": 7.2,
              "Australia": 8.9}
    ends = {"US": 4.2, "Germany": 2.7, "Japan": 1.6, "Britain": 4.5, "Canada": 3.3,
            "Australia": 4.4}
    fig = go.Figure()
    for name in starts:
        path = np.linspace(starts[name], ends[name], len(_YEARS)) + rng.normal(0, 0.18, len(_YEARS))
        fig.add_trace(go.Scatter(x=_YEAR_DATES, y=np.round(path, 2), name=name))
    highlight(fig, "US")
    economist(
        fig,
        title="The climb back down",
        subtitle="Ten-year government-bond yields, %",
        source="Source: FRED",
    )
    _apply_year_ticks(fig)
    return fig


def fig_step_with_periods() -> go.Figure:
    """Flagship: fed funds step line with recession shading and event arrows.

    Returns:
        Styled step figure.
    """
    dates, levels = zip(*_FED_FUNDS, strict=True)
    fig = go.Figure(
        go.Scatter(
            x=list(dates), y=list(levels), line=dict(color=MAIN["BLUE"], width=2.2), line_shape="hv"
        )
    )
    economist(
        fig,
        title="Three decades of rate decisions",
        subtitle="United States, effective federal funds rate, %",
        source="Source: FRED; The Economist",
        footnote="* Shaded areas mark recessions",
        legend=False,
    )
    fig.update_yaxes(range=[0, 7], tick0=0, dtick=1)
    _apply_year_ticks(fig)
    period_shading(
        fig,
        [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")],
        labels=["Dot-com bust", "Financial crisis", "Covid-19"],
    )
    annotate_event(fig, "2008-12-16", "Zero lower bound", ax=-46, ay=-52)
    annotate_event(fig, "2023-07-27", "Fastest hiking on record", ax=-70, ay=-38)
    return fig


def fig_signed_bars() -> go.Figure:
    """Annual rate changes in basis points, hikes against cuts.

    Returns:
        Styled signed column figure.
    """
    years, values = zip(*_RATE_CHANGES, strict=True)
    fig = go.Figure(
        go.Bar(x=[str(y) for y in years], y=values, marker=dict(color=signed(values)))
    )
    economist(
        fig,
        title="From hikes to cuts",
        subtitle="Change in the federal funds target rate during the year, basis points",
        source="Source: FRED",
        legend=False,
    )
    return fig


def fig_matplotlib() -> Path:
    """Single CPI line through the matplotlib escape hatch.

    Returns:
        Path to the written PNG.
    """
    with mpl_style():
        fig = plt.figure(figsize=(332 / 72, 238.8 / 72))
        ax = fig.add_subplot()
        ax.plot(_YEARS, _CPI, color=MAIN["BLUE"], linewidth=1.0)
        ax.set_ylim(-1, 9)
        ax.set_xlim(1994.4, 2025.6)
        ticks = [1995, 2000, 2005, 2010, 2015, 2020, 2025]
        labels = ["1995"] + [f"{y % 100:02d}" for y in ticks[1:]]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(range(-1, 10, 2))
        fig.subplots_adjust(left=0.03, right=0.92, top=0.79, bottom=0.14)
        mpl_furniture(
            ax,
            title="Inflation is back near target",
            subtitle="United States, consumer prices, % change on a year earlier",
            source="Source: FRED",
        )
        path = OUT / "06_matplotlib.png"
        fig.savefig(path, dpi=258.2)
        plt.close(fig)
    return path


def main() -> list[Path]:
    """Render all example figures.

    Returns:
        Paths of the written PNGs.
    """
    register()
    figs = {
        "01_single_line.png": fig_single_line(),
        "02_multi_line.png": fig_multi_line(),
        "03_highlight.png": fig_highlight(),
        "04_step_with_periods.png": fig_step_with_periods(),
        "05_signed_bars.png": fig_signed_bars(),
    }
    paths = []
    for name, fig in figs.items():
        path = save(fig, OUT / name, size="col2", scale=2)
        paths.append(path)
    paths.append(fig_matplotlib())
    paths.append(save(fig_multi_line(size="slide"), OUT / "07_slide.png", size="slide", scale=2))
    for path in paths:
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
