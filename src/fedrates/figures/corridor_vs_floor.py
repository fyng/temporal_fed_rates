"""Production chart: corridor versus floor, a schematic contrast of regimes.

Run:
    uv run python -m fedrates.figures.corridor_vs_floor
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedrates import register, save
from fedrates.theme import (
    BLACK,
    FONT_BODY,
    MAIN,
    SCALES,
    SIZES,
    TEXT,
    economist,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"

TITLE = "Two ways to set a price"
SUBTITLE = (
    "Schematic: interest rate against the quantity of reserves, before and after October 2008"
)
SOURCE = "Source: Federal Reserve"
FOOTNOTE = (
    "Interest on reserves arrived in October 2008; ample reserves adopted on January 30th 2019"
)

_XR = [0, 10]
_YR = [0, 7.2]
_SUPPLY = SCALES["GREY"][1]
_ARROW = "rgba(12,12,12,0.5)"

# Steep scarce-reserve demand: y = 6.2 - 1.2619 * (x - 0.6).
_DEMAND_SLOPE = (0.9 - 6.2) / (4.8 - 0.6)
_A1 = (2.0, 6.2 + _DEMAND_SLOPE * (2.0 - 0.6))  # operating point on S
_A2 = (3.2, 6.2 + _DEMAND_SLOPE * (3.2 - 0.6))  # operating point on S'


def _line(x: list[float], y: list[float], color: str, width: float = 2.2, dash: str | None = None):
    """Build a schematic line trace.

    Args:
        x: x coordinates.
        y: y coordinates.
        color: Line colour.
        width: Line width in px.
        dash: Dash style, solid when None.

    Returns:
        Scatter trace with hover disabled.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=color, width=width, dash=dash or "solid"),
        hoverinfo="skip",
        showlegend=False,
    )


def _note(
    fig: go.Figure,
    axis: Literal["x", "x2"],
    x: float,
    y: float,
    text: str,
    *,
    ax: float = 0,
    ay: float = 0,
    align: str = "center",
    arrow: bool = False,
) -> None:
    """Add a labelled note to one schematic panel.

    Args:
        fig: Figure to annotate.
        axis: Panel x-axis id ("x" or "x2").
        x: Anchor x in data coordinates.
        y: Anchor y in data coordinates.
        text: Label text; <br> splits lines.
        ax: Horizontal text offset in px.
        ay: Vertical text offset in px; negative is above.
        align: Horizontal anchoring of the text box at its offset position.
        arrow: Draw a leader arrow to the anchor.
    """
    fig.add_annotation(
        x=x,
        y=y,
        xref=axis,
        yref=f"y{axis[1:]}",
        ax=ax,
        ay=ay,
        text=text,
        showarrow=arrow,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=0.9,
        arrowcolor=_ARROW,
        standoff=3 if arrow else 0,
        align=align,
        xanchor=align,
        font=dict(family=FONT_BODY, size=12.5, color=TEXT),
    )


def _corridor(fig: go.Figure) -> None:
    """Draw the pre-2008 corridor panel: the Desk steers quantity.

    Args:
        fig: Figure whose first axes receive the traces.
    """
    fig.add_trace(_line([0.6, 4.8], [6.2, 0.9], MAIN["BLUE"]), row=1, col=1)
    fig.add_trace(_line([0, 10], [6.6, 6.6], MAIN["GREY"], width=1.2, dash="dash"), row=1, col=1)
    fig.add_trace(_line([2.0, 2.0], [0.5, 6.6], _SUPPLY, width=1.4), row=1, col=1)
    fig.add_trace(_line([3.2, 3.2], [0.5, 6.6], _SUPPLY, width=1.4), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[_A1[0], _A2[0]],
            y=[_A1[1], _A2[1]],
            mode="markers",
            marker=dict(color=BLACK, size=7),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _note(fig, "x", 3.2, _A2[1], "The Desk moves the supply<br>of reserves to set the rate",
          ax=127, ay=-40, arrow=True)
    _note(fig, "x", 0.12, 6.9, "Discount rate (ceiling)", align="left")
    _note(fig, "x", 1.88, 6.25, "S", align="right")
    _note(fig, "x", 3.32, 6.25, "S'", align="left")
    _note(fig, "x", 4.3, 1.7, "Demand for reserves", align="left")
    _note(fig, "x", 1.85, _A1[1], "Target", align="right")
    _note(fig, "x", 0.12, 0.32, "Scarce reserves", align="left")
    _note(
        fig,
        "x",
        9.75,
        5.85,
        "Reserves earned no interest<br>"
        "before 2008, so demand is steep<br>"
        "and quantity sets the rate",
        align="right",
    )


def _floor(fig: go.Figure) -> None:
    """Draw the post-2008 floor panel: the Fed announces prices.

    Args:
        fig: Figure whose second axes receive the traces.
    """
    fig.add_trace(_line([0.6, 2.5], [6.2, 1.0], MAIN["BLUE"]), row=1, col=2)
    fig.add_trace(_line([2.5, 9.5], [1.0, 1.0], MAIN["BLUE"]), row=1, col=2)
    fig.add_trace(_line([0, 10], [1.2, 1.2], MAIN["GREEN"], width=1.2, dash="dash"), row=1, col=2)
    fig.add_trace(_line([0, 10], [0.75, 0.75], MAIN["CYAN"], width=1.2, dash="dash"), row=1, col=2)
    fig.add_trace(_line([5.0, 5.0], [0.4, 3.2], _SUPPLY, width=1.4), row=1, col=2)
    fig.add_trace(_line([7.0, 7.0], [0.4, 3.2], _SUPPLY, width=1.4), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=[5.0, 7.0],
            y=[1.0, 1.0],
            mode="markers",
            marker=dict(color=BLACK, size=7),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _note(fig, "x2", 6.0, 1.32, "Moving the quantity<br>no longer moves the rate",
          ay=-189, arrow=True)
    _note(fig, "x2", 0.12, 1.42, "IORB", align="left")
    _note(fig, "x2", 0.12, 0.48, "ON RRP", align="left")
    _note(fig, "x2", 1.75, 3.6, "Demand for reserves", align="left")
    _note(fig, "x2", 7.6, 0.32, "Ample reserves", align="left")


def _panel_titles(fig: go.Figure) -> None:
    """Add bold lowercase panel letters with terse regime titles.

    Args:
        fig: Figure to annotate.
    """
    for axis, text in (
        ("x", "<b>a</b>&nbsp;&nbsp;Before 2008: steer the quantity"),
        ("x2", "<b>b</b>&nbsp;&nbsp;Since 2008: announce prices"),
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
            font=dict(family=FONT_BODY, size=13, color=TEXT),
        )


def _y_titles(fig: go.Figure) -> None:
    """Put one rotated axis label to the left of each panel.

    Args:
        fig: Figure with two side-by-side schematic panels.
    """
    for x in (-0.011, 0.492):  # left of panel (a); the gap between panels
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x,
            y=0.5,
            text="Interest rate",
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(family=FONT_BODY, size=12.5, color=TEXT),
        )


def build_fig() -> go.Figure:
    """Build the corridor-versus-floor schematic.

    Returns:
        Decorated Economist figure.
    """
    register()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10)
    _corridor(fig)
    _floor(fig)
    _panel_titles(fig)
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
    fig.update_xaxes(range=_XR, ticks="", showticklabels=False)
    fig.update_yaxes(range=_YR, showgrid=False, ticks="", showticklabels=False)
    _y_titles(fig)
    # Room for the rotated labels; keep the brand tag flush to the corner.
    fig.update_layout(margin=dict(l=26))
    pw = SIZES["col3"]["px"][0] - 26 - 44  # plot width after the new left margin
    tag = fig.layout.shapes[0]  # the red tag, the only shape economist() adds
    tag.x0 = -26 / pw
    tag.x1 = tag.x0 + 15 / pw
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("corridor_vs_floor.png", "corridor_vs_floor.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
