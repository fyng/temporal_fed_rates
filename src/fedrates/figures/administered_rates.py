"""Production chart: administered rates and the effective rate inside the band.

Run:
    uv run python -m fedrates.figures.administered_rates
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedrates import fomc, fred, register, save, transforms
from fedrates.theme import (
    BLACK,
    ECON_RED,
    FONT_BODY,
    GREY_LABEL,
    MAIN,
    TEXT,
    economist,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"
_START = pd.Timestamp("2013-09-01")
_END = pd.Timestamp("2026-10-01")
_IOR_SPLICE = "2021-07-29"  # IOER ends 2021-07-28; IORB begins 2021-07-29

TITLE = "Inside the band"
SUBTITLE = "United States, daily policy rates, %"
SOURCE = "Source: Federal Reserve; FRED"
FOOTNOTE = "Rate on reserves is IOER to July 2021, IORB thereafter"

_XR = [_START, _END]
_TICKVALS = [f"{y}-01-01" for y in range(2014, 2027, 2)]
_TICKTEXT = ["2014", "16", "18", "20", "22", "24", "26"]
_BAND_FILL = "rgba(164,189,201,0.22)"
_ARROW = "rgba(12,12,12,0.5)"

# Detail box over a flat, well-separated window: ON RRP 5.30 < EFFR 5.33 < IORB 5.40.
_INSET_X = ["2023-12-20", "2024-09-10"]
_INSET_Y = [5.24, 5.46]
_INSET_TICKVALS = ["2024-01-01", "2024-04-01", "2024-07-01"]
_INSET_TICKTEXT = ["Jan 2024", "Apr", "Jul"]
_INSET_YTICKVALS = [5.30, 5.35, 5.40]
_INSET_YTICKTEXT = ["5.30", "", "5.40"]


def _band(meetings: pd.DataFrame) -> pd.DataFrame:
    """Target range bounds over the chart window as a step series.

    Args:
        meetings: Committed FOMC decision record.

    Returns:
        Frame of date, target_lower, target_upper, carried to the window edges.
    """
    prior = meetings.loc[meetings["date"] < _START].iloc[-1]
    in_win = meetings.loc[meetings["date"] >= _START, ["date", "target_lower", "target_upper"]]
    head = pd.DataFrame(
        {
            "date": [_START],
            "target_lower": [prior["target_lower"]],
            "target_upper": [prior["target_upper"]],
        }
    )
    tail = pd.DataFrame(
        {
            "date": [_END],
            "target_lower": [in_win["target_lower"].iloc[-1]],
            "target_upper": [in_win["target_upper"].iloc[-1]],
        }
    )
    return pd.concat([head, in_win, tail], ignore_index=True)


def _rate_trace(x: pd.Series, y: pd.Series, name: str, color: str, width: float) -> go.Scatter:
    """Build a daily-rate line trace.

    Args:
        x: Dates.
        y: Rates.
        name: Legend label.
        color: Line colour.
        width: Line width in px.

    Returns:
        Scatter trace bridging holiday gaps.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        connectgaps=True,
    )


def _inset_trace(x: pd.Series, y: pd.Series, color: str) -> go.Scatter:
    """Build a line trace bound to the detail-box axes.

    Args:
        x: Dates.
        y: Rates.
        color: Line colour.

    Returns:
        Scatter trace on xaxis3/yaxis3, outside the legend.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        xaxis="x3",
        yaxis="y3",
        line=dict(color=color, width=2.0),
        connectgaps=True,
        hoverinfo="skip",
        showlegend=False,
    )


def _leader(
    fig: go.Figure,
    xref: str,
    yref: str,
    x,
    y: float,
    text: str,
    *,
    ax: float,
    ay: float,
) -> None:
    """Add a leader annotation in axis coordinates.

    Args:
        fig: Figure to annotate.
        xref: x-axis reference, "x2" for the spread panel.
        yref: y-axis reference, "y2" for the spread panel.
        x: Anchor x.
        y: Anchor y.
        text: Label text; <br> splits lines.
        ax: Horizontal text offset in px.
        ay: Vertical text offset in px; negative is above.
    """
    fig.add_annotation(
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        ax=ax,
        ay=ay,
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=0.9,
        arrowcolor=_ARROW,
        standoff=3,
        align="center",
        font=dict(family=FONT_BODY, size=12.5, color=TEXT),
    )


def _panel_notes(fig: go.Figure, spread: pd.Series) -> None:
    """Add panel letters, the repo-spike callout and the closing spread value.

    Args:
        fig: Figure to annotate.
        spread: EFFR minus the rate on reserves, in basis points.
    """
    for axis, text in (
        ("x", "<b>a</b>&nbsp;&nbsp;Policy rates"),
        ("x2", "<b>b</b>&nbsp;&nbsp;EFFR minus rate on reserves, basis points"),
    ):
        fig.add_annotation(
            x=0,
            y=1,
            xref=f"{axis} domain",
            yref=f"y{axis[1:]} domain",
            xanchor="left",
            yanchor="bottom",
            yshift=9,
            text=text,
            showarrow=False,
            font=dict(family=FONT_BODY, size=12.5, color=TEXT),
        )
    spike = spread.loc["2019-09-01":"2019-10-01"]
    _leader(
        fig,
        "x2",
        "y2",
        spike.idxmax(),
        float(spike.max()),
        "Repo spike,<br>September 17th 2019",
        ax=-110,
        ay=-8,
    )
    _leader(
        fig,
        "x2",
        "y2",
        spread.index[-1],
        float(spread.iloc[-1]),
        f"{spread.iloc[-1]:.0f}bp",
        ax=-44,
        ay=-30,
    )


def build_fig() -> go.Figure:
    """Build the administered-rates chart.

    Returns:
        Decorated Economist figure with levels above and the spread below.
    """
    register()
    meetings = fomc.load_meetings()
    band = _band(meetings)
    effr = fred.fetch("EFFR").loc[_START:]
    ior = transforms.splice(fred.fetch("IORB"), fred.fetch("IOER"), at=_IOR_SPLICE)
    ior = ior.loc[_START:]
    onrrp = fred.fetch("RRPONTSYAWARD").loc[_START:]
    spread = ((effr - ior) * 100.0).dropna()

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08, row_heights=[0.68, 0.32])
    fig.add_trace(
        go.Scatter(
            x=band["date"],
            y=band["target_upper"],
            mode="lines",
            name=f"Target range {band['target_lower'].iloc[-1]:.2f}-"
            f"{band['target_upper'].iloc[-1]:.2f}",
            line=dict(color=GREY_LABEL, width=0.75),
            line_shape="hv",
            legendrank=1,
            legendgroup="band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=band["date"],
            y=band["target_lower"],
            mode="lines",
            line=dict(color=GREY_LABEL, width=0),
            fill="tonexty",
            fillcolor=_BAND_FILL,
            line_shape="hv",
            legendgroup="band",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _rate_trace(onrrp.index, onrrp, f"ON RRP {onrrp.iloc[-1]:.2f}", MAIN["CYAN"], 1.6),
        row=1,
        col=1,
    )
    fig.add_trace(
        _rate_trace(ior.index, ior, f"Rate on reserves {ior.iloc[-1]:.2f}", MAIN["GREEN"], 1.6),
        row=1,
        col=1,
    )
    fig.add_trace(
        _rate_trace(effr.index, effr, f"EFFR {effr.iloc[-1]:.2f}", MAIN["BLUE"], 2.2),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.to_numpy(),
            mode="lines",
            line=dict(color=MAIN["BLUE"], width=1.6),
            connectgaps=True,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for y, color in ((onrrp, MAIN["CYAN"]), (ior, MAIN["GREEN"]), (effr, MAIN["BLUE"])):
        window = (y.index >= pd.Timestamp(_INSET_X[0])) & (y.index <= pd.Timestamp("2024-08-31"))
        fig.add_trace(_inset_trace(y.index[window], y[window], color))
    ranks = {"band": 1, fig.data[4].name: 2, fig.data[3].name: 3, fig.data[2].name: 4}

    _panel_notes(fig, spread)
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
    for row in (1, 2):
        fig.update_xaxes(range=_XR, tickvals=_TICKVALS, ticktext=_TICKTEXT, row=row, col=1)
    fig.update_xaxes(row=1, col=1, showticklabels=False)
    # Legend swatches are carrier traces added by the theme; rank them so the
    # order reads band, EFFR, IOR, ON RRP.
    for trace in fig.data:
        if trace.showlegend and trace.x is not None and len(trace.x) == 1 and trace.x[0] is None:
            trace.legendrank = ranks.get(trace.legendgroup, 1000)
    fig.update_yaxes(range=[0, 6.2], dtick=1, row=1, col=1)
    fig.update_yaxes(
        range=[-36, 28],
        dtick=10,
        zeroline=True,
        zerolinecolor=ECON_RED,
        zerolinewidth=0.72,
        row=2,
        col=1,
    )
    fig.update_xaxes(row=2, col=1, showline=False, ticks="")
    fig.update_layout(
        xaxis3=dict(
            domain=[0.035, 0.315],
            range=_INSET_X,
            anchor="y3",
            type="date",
            tickvals=_INSET_TICKVALS,
            ticktext=_INSET_TICKTEXT,
            showgrid=False,
            showline=True,
            linecolor=BLACK,
            linewidth=0.72,
            mirror=True,
            ticks="outside",
            ticklen=3,
            tickcolor=BLACK,
            tickfont=dict(family=FONT_BODY, size=11, color=TEXT),
        ),
        yaxis3=dict(
            domain=[0.76, 0.95],
            range=_INSET_Y,
            anchor="x3",
            side="right",
            tickvals=_INSET_YTICKVALS,
            ticktext=_INSET_YTICKTEXT,
            showgrid=False,
            showline=True,
            linecolor=BLACK,
            linewidth=0.72,
            mirror=True,
            ticks="",
            tickfont=dict(family=FONT_BODY, size=11, color=TEXT),
        ),
    )
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("administered_rates.png", "administered_rates.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
