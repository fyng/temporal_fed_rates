"""Production chart: who does what, from FOMC decision to overnight market.

Run:
    uv run python -m fedrates.figures.who_does_what
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from fedrates import register, save
from fedrates.theme import (
    BLACK,
    FONT_BODY,
    MAIN,
    MUTED,
    PANEL,
    PT_TO_PX,
    TEXT,
    economist,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"

TITLE = "Who does what"
SUBTITLE = "How a Fed rate decision reaches the overnight market, September 2026"
SOURCE = "Sources: Federal Reserve; Federal Reserve Bank of New York"

_XR, _YR = [0, 10], [1.6, 10.0]
_ARROW = "rgba(12,12,12,0.55)"
_FED, _MKT = MAIN["BLUE"], MAIN["YELLOW"]

# (x0, x1, y0, y1, main, sub, side); y0 is the top edge, y1 the bottom edge.
# sub is a small muted line (FOMC) or a bold rate (facilities).
_BOXES = [
    (2.2, 7.8, 8.5, 9.7, "Federal Open Market Committee",
     "Sets the target range and the standing repo rates", "fed"),
    (3.4, 6.6, 6.75, 7.55, "New York Fed trading desk", None, "fed"),
    (0.3, 2.5, 4.65, 5.75, "Interest on reserves", "3.90%", "fed"),
    (2.7, 4.9, 4.65, 5.75, "Reverse repo", "3.75%", "fed"),
    (5.3, 7.5, 4.65, 5.75, "Standing repo", "4.00%", "fed"),
    (7.7, 9.9, 4.65, 5.75, "Discount window", "4.00%", "fed"),
    (0.35, 2.35, 2.6, 3.5, "Federal Home Loan Banks", None, "mkt"),
    (2.75, 4.95, 2.6, 3.5, "Foreign bank branches", None, "mkt"),
    (5.25, 7.45, 2.6, 3.5, "Money-market funds", None, "mkt"),
    (7.75, 9.95, 2.6, 3.5, "Banks", None, "mkt"),
]

# (x0, y0, x1, y1, label, lx, ly, anchor, dash, width); the label, when set,
# sits at (lx, ly) with the given anchoring and describes the arrow.
_ARROWS = [
    (5.0, 8.5, 5.0, 7.55, "Target range 3.75-4.00%", 5.2, 8.02, "left", None, 1.2),
    (5.0, 6.75, 5.0, 5.8, "Implements the rates", 5.2, 6.27, "left", None, 1.2),
    (2.35, 3.05, 2.75, 3.05, "Fed funds", 2.55, 3.62, "center", None, 1.2),
    (3.85, 3.5, 2.2, 4.75, "Park reserves", 1.75, 4.35, "right", None, 1.2),
    (6.35, 3.5, 4.5, 4.75, "Deposit cash", 4.85, 4.15, "right", None, 1.2),
    (8.15, 3.5, 5.7, 4.75, "Borrow in a pinch", 8.45, 4.2, "center", "dash", 1.2),
    (9.35, 3.5, 9.5, 4.75, None, 0.0, 0.0, "left", "dash", 1.2),
    (1.0, 3.5, 2.7, 5.0, None, 0.0, 0.0, "left", "dash", 1.2),
]

# Which body sets each administered rate.
_SET_BY = {
    "Interest on reserves": "Set by the Board",
    "Reverse repo": "Set by the FOMC",
    "Standing repo": "Set by the FOMC",
    "Discount window": "Set by the Board",
}

_NOTE = ("Effective rate 3.88%, published daily by the New York Fed", 2.55, 1.95)

_GROUPS = [("FLOORS", 2.6), ("CEILINGS", 7.6)]  # label, x centre over the row


def _box(fig: go.Figure, x0, x1, y0, y1, main, sub, side) -> None:
    """Add a PANEL-filled box with a coloured top bar and centred text.

    Args:
        fig: Figure to draw on.
        x0: Left edge in data units.
        x1: Right edge in data units.
        y0: Bottom edge in data units.
        y1: Top edge in data units.
        main: Primary label.
        sub: Small muted line, bold rate, or None for a single-line box.
        side: "fed" or "mkt", choosing the accent colour.
    """
    accent = _FED if side == "fed" else _MKT
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=PANEL, line_width=0)
    fig.add_shape(type="rect", x0=x0, y0=y1 - 0.07, x1=x1, y1=y1,
                  fillcolor=accent, line_width=0)
    cx, pad = (x0 + x1) / 2, 0.3
    body = dict(family=FONT_BODY, color=TEXT)
    if sub is None:
        fig.add_annotation(x=cx, y=(y0 + y1) / 2, text=main, showarrow=False,
                           font=body | dict(size=11))
    elif sub.endswith("%"):
        fig.add_annotation(x=cx, y=y1 - pad, text=main, showarrow=False,
                           font=body | dict(size=10.5))
        fig.add_annotation(x=cx, y=y0 + pad, text=f"<b>{sub}</b>", showarrow=False,
                           font=body | dict(size=12.5))
        fig.add_annotation(x=cx, y=y0 + 0.06, text=_SET_BY[main], showarrow=False,
                           font=body | dict(size=8, color=MUTED))
    else:
        fig.add_annotation(x=cx, y=y1 - pad, text=main, showarrow=False,
                           font=body | dict(size=12))
        fig.add_annotation(x=cx, y=y0 + pad, text=sub, showarrow=False,
                           font=body | dict(size=10.5, color=MUTED))


def _arrows(fig: go.Figure, sx: float, sy: float) -> None:
    """Draw the schematic arrows and their labels.

    Args:
        fig: Figure to draw on.
        sx: Plot pixels per x data unit.
        sy: Plot pixels per y data unit.
    """
    for x0, y0, x1, y1, label, lx, ly, anchor, dash, width in _ARROWS:
        ax, ay = (x0 - x1) * sx, -(y0 - y1) * sy
        size, so = (1.3, 0) if dash else (0.9, 1)
        if dash:  # annotation arrows are solid: line shape for the shaft
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          xref="x", yref="y",
                          line=dict(color=_ARROW, width=width, dash=dash))
            norm = (ax * ax + ay * ay) ** 0.5
            # plotly drops heads on arrows shorter than ~5px: keep an 8px tail
            ax, ay = 8.0 * ax / norm, 8.0 * ay / norm
        fig.add_annotation(
            x=x1, y=y1, ax=ax, ay=ay,
            xref="x", yref="y", showarrow=True, arrowhead=2, arrowsize=size,
            arrowwidth=width, arrowcolor=BLACK if width > 2 else _ARROW, standoff=so,
        )
        if label:
            fig.add_annotation(x=lx, y=ly, xanchor=anchor, text=label, showarrow=False,
                               font=dict(family=FONT_BODY, size=10.5, color=TEXT))


def _scales(fig: go.Figure) -> tuple[float, float]:
    """Return plot pixels per x and y data unit.

    Args:
        fig: Decorated figure with final margins.

    Returns:
        (sx, sy) in pixels per data unit.
    """
    m = fig.layout.margin
    pw = fig.layout.width - m.l - m.r
    ph = fig.layout.height - m.t - m.b
    return pw / (_XR[1] - _XR[0]), ph / (_YR[1] - _YR[0])


def build_fig() -> go.Figure:
    """Build the who-does-what schematic.

    Returns:
        Decorated Economist figure.
    """
    register()
    fig = go.Figure()
    for name, color in (("Federal Reserve", _FED), ("Market participants", _MKT)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", color=color),
                                 showlegend=True, hoverinfo="skip"))
    for box in _BOXES:
        _box(fig, *box)
    economist(fig, title=TITLE, subtitle=SUBTITLE, source=SOURCE, size=SIZE,
              legend=True, zeroline=False)
    fig.update_xaxes(range=_XR, visible=False)
    fig.update_yaxes(range=_YR, visible=False)
    _arrows(fig, *_scales(fig))
    for text, gx in _GROUPS:  # uppercase period-style labels over the row
        fig.add_annotation(x=gx, y=6.0, text=text, showarrow=False,
                           font=dict(family=FONT_BODY, weight=300,
                                     size=round(6.5 * PT_TO_PX, 1), color=TEXT))
    text, nx, ny = _NOTE
    fig.add_annotation(x=nx, y=ny, text=text, showarrow=False,
                       font=dict(family=FONT_BODY, size=10, color=MUTED))
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("who_does_what.png", "who_does_what.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
