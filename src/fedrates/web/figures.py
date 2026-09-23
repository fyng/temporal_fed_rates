"""Interactive web figures for the single-page HTML article.

Each builder returns a responsive plotly figure carrying all of its
interactivity inside the figure JSON (updatemenus, rangeselector, hover
templates, legend toggling); no custom JavaScript. The HTML build script
renders title, subtitle and source around each chart, so the figures carry no
title block, source line or brand tag.

Run ``python -m fedrates.web.figures`` to write inspection copies to
``figures/explore/web/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedrates import dataset, fomc, fred, register, transforms
from fedrates.figures.who_does_what import _ARROWS, _BOXES, _GROUPS, _NOTE, _SET_BY
from fedrates.models import decision, level
from fedrates.theme import (
    BLACK,
    ECON_RED,
    FONT_BODY,
    GREY_LABEL,
    MAIN,
    MUTED,
    PANEL,
    RULE,
    SCALES,
    TEXT,
    _font,
    _text_width,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "explore" / "web"
ABLATION_CSV = Path(__file__).resolve().parents[3] / "data" / "processed" / "decision_ablation.csv"
CUT25_CSV = Path(__file__).resolve().parents[3] / "data" / "processed" / "cut25_walk_forward.csv"

WEB_WIDTH = 640  # assumed article-column width, for pixel-anchored schematics
_PAD = 8.0  # px side pad
_Y_ROOM = 44.0  # px reserved for right-hand y tick labels
_LEG_BAND = 30.0  # px top margin for an inline legend
_X_BAND = 26.0  # px bottom margin for x tick labels
_ARROW = "rgba(12,12,12,0.5)"  # house annotation arrow, 50% black
_CHAIR_RULE = RULE
_REC_FILL = "rgba(12,12,12,0.14)"
_H_WHO = 560
_GROUP_ENDS = frozenset({"market", "full", "macro"})

_ROWS = ("1994+ Q", "1994+ M", "1983+ Q", "1983+ M", "1961+ Q", "1961+ M")
_FREQ = {"M": ("Monthly", MAIN["GREY"]), "Q": ("Quarterly", MAIN["BLUE"])}
_FREQ_NAME = {"M": "monthly", "Q": "quarterly"}
_SPEC_GRID_XR = {"a": [-3.6, 6.0], "b": [-1.0, 3.4]}
_SPEC_GRID_DTICK = {"a": 2, "b": 1}

ABL_ROWS: tuple[tuple[str, str], ...] = (
    ("history+market", "History + market rates"),
    ("market", "Market rates only"),
    ("history", "History only"),
    ("history+rule", "History + rule gaps"),
    ("history+macro", "History + economy"),
    ("full", "Everything"),
    ("rule", "Rule gaps only"),
    ("macro", "Economy only"),
    ("constant_hold", "Always hold"),
)
ABL_FOCUS = "history+market"
ABL_PANELS: tuple[tuple[str, str, tuple[float, float], tuple[float, ...]], ...] = (
    ("rps", "Probability score", (0, 0.124), (0, 0.1)),
    ("auroc_macro", "ROC area", (0, 1.02), (0, 1)),
    ("auprc_macro", "Precision-recall area", (0, 1.02), (0, 1)),
    ("f1_macro", "F1 score", (0, 1.02), (0, 1)),
)

_BOX_NOTES = {
    "Federal Open Market Committee": (
        "Sets the target range and the standing repo rates eight times a year. "
        "Twelve votes: seven governors, the New York Fed president and four "
        "rotating regional presidents."
    ),
    "New York Fed trading desk": (
        "Carries the decisions out from the trading floor: manages the supply of "
        "reserves and administers the Fed's standing facilities."
    ),
    "Interest on reserves": (
        "Interest on reserves, the floor: banks will not lend overnight for less "
        "than this administered rate. Rate set by the Board."
    ),
    "Reverse repo": (
        "Reverse repo, the floor for cash that cannot reach reserves: money funds "
        "and other counterparties lend to the Fed against Treasuries overnight. "
        "Rate set by the FOMC."
    ),
    "Standing repo": (
        "Standing repo, the ceiling: dealers can always borrow against Treasuries "
        "at this rate. Rate set by the FOMC."
    ),
    "Discount window": (
        "Discount window, the ceiling for banks in need of liquidity; borrowing is "
        "stigma-tinged and rare. Rate set by the Board."
    ),
    "Federal Home Loan Banks": (
        "Government-sponsored lenders that channel cash from their members into the "
        "fed funds market; also eligible to lend at the reverse repo facility."
    ),
    "Foreign bank branches": (
        "Large holders of reserves; active borrowers and lenders in the overnight market."
    ),
    "Money-market funds": (
        "Cash pools that cannot earn interest on reserves; they use the reverse repo "
        "facility instead."
    ),
    "Banks": "The largest borrowers and lenders in the overnight market.",
}


# --- Shared helpers -------------------------------------------------------------


def _base(
    fig: go.Figure,
    *,
    height: int,
    hovermode: Literal["x unified", "closest"] = "x unified",
    top: float = _LEG_BAND,
    left: float = _PAD,
    right: float = _Y_ROOM,
    bottom: float = _X_BAND,
    legend: bool | Literal["right"] = True,
    ytitle: str | None = None,
    xtitle: str | None = None,
) -> go.Figure:
    """Apply the web layout: responsive width, fixed height, top or right legend.

    Args:
        fig: Figure to style.
        height: Fixed pixel height.
        hovermode: Plotly hover mode.
        top: Top margin in px.
        left: Left margin in px.
        right: Right margin in px.
        bottom: Bottom margin in px.
        legend: True for a horizontal legend above the plot, "right" for a
            single-column legend to the right, False for none.
        ytitle: Optional y-axis title; units also ride in hover values.
        xtitle: Optional x-axis title.

    Returns:
        The styled figure.
    """
    fig.update_layout(
        autosize=True,
        width=None,
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        hovermode=hovermode,
    )
    if ytitle:
        fig.update_yaxes(title=dict(text=ytitle, font=_font("tick"), standoff=8))
    if xtitle:
        fig.update_xaxes(title=dict(text=xtitle, font=_font("tick"), standoff=6))
    if legend is False:
        fig.update_layout(showlegend=False)
    elif legend == "right":
        fig.update_layout(
            legend=dict(
                orientation="v",
                xref="paper",
                yref="paper",
                x=1.08,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=_font("tick"),
                itemsizing="constant",
            )
        )
    else:
        fig.update_layout(
            legend=dict(
                orientation="h",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=_font("tick"),
                tracegroupgap=4,
                itemsizing="constant",
            )
        )
    return fig


def _menu(
    fig: go.Figure,
    buttons: list[dict],
    *,
    dropdown: bool = False,
    x: float = 0.0,
    y: float = 1.0,
    active: int = 0,
) -> None:
    """Add a theme-styled updatemenu: PANEL background, TEXT font, no border.

    Args:
        fig: Figure to attach the menu to.
        buttons: Plotly button dicts.
        dropdown: Render as a dropdown rather than a button row.
        x: Menu x position in paper coordinates.
        y: Menu y position in paper coordinates.
        active: Index of the initially active button.
    """
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown" if dropdown else "buttons",
                direction="down" if dropdown else "right",
                x=x,
                y=y,
                xanchor="left",
                yanchor="bottom",
                bgcolor=PANEL,
                bordercolor=RULE,
                borderwidth=0,
                font=_font("tick"),
                pad=dict(l=2, r=2, t=1, b=1),
                buttons=buttons,
                active=active,
                showactive=True,
            )
        ]
    )


def _stack_top(fig: go.Figure, *, height: int, top: float, bottom: float = _X_BAND) -> None:
    """Put the legend at the top edge and the menu row just above the plot.

    A legend that wraps on a narrow screen pushes the plot down, and the menu,
    anchored to the plot, moves down with it instead of overlapping the legend.

    Args:
        fig: Figure with one updatemenu and a horizontal legend.
        height: Figure height in px.
        top: Top margin in px; room for one legend row plus the menu.
        bottom: Bottom margin in px.
    """
    plot_h = height - top - bottom
    fig.layout.updatemenus[0].update(y=1 + 6 / plot_h, yanchor="bottom")
    fig.update_layout(legend=dict(yref="container", y=1 - 2 / height, yanchor="top"))


def _period_label(fig: go.Figure, x, text: str, *, angle: int = 0) -> None:
    """Add an uppercase period-style label at the top of the plot.

    Args:
        fig: Figure to annotate.
        x: Anchor in x-axis coordinates.
        text: Label text, rendered uppercase.
        angle: Text angle for narrow spans.
    """
    fig.add_annotation(
        xref="x",
        yref="paper",
        x=x,
        y=1,
        yanchor="top",
        yshift=-2,
        textangle=angle,
        text=str(text).upper(),
        showarrow=False,
        font=_font("period"),
    )


def _chair_spans(end) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Chair tenures clipped to a chart window.

    Args:
        end: Chart end date; opens the sitting chair's span.

    Returns:
        (chair, start, end) triples in chronological order.
    """
    out = []
    for name, start, stop in fomc.CHAIRS:
        e = pd.Timestamp(stop) if stop else pd.Timestamp(end)
        out.append((name, pd.Timestamp(start), e))
    return out


def _recession_spans() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """NBER recession spans with ends extended to full months.

    Returns:
        (start, end) pairs shaped for shading rectangles.
    """
    return [
        (s, e + pd.offsets.MonthBegin(1))
        for s, e in dataset.recession_spans(dataset.load_frame())
    ]


def _shade(fig: go.Figure, spans, fill: str, *, yref: str = "paper",
           layer: str = "below") -> None:
    """Shade date spans behind the data.

    Args:
        fig: Figure to decorate.
        spans: (start, end) date pairs.
        fill: Rectangle fill colour.
        yref: Shape y reference, e.g. "paper" or a subplot "y domain".
        layer: Plotly shape layer.
    """
    for x0, x1 in spans:
        fig.add_shape(
            type="rect",
            xref="x",
            yref=yref,
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=fill,
            line_width=0,
            layer=layer,
        )


def _target_text(row) -> str:
    """Target range or point in percent, for hover text.

    Args:
        row: Meetings row with target columns.

    Returns:
        Text such as "3.75-4.00%" or "5.25%".
    """
    if pd.notna(row["target"]):
        return f"{row['target']:.2f}%"
    return f"{row['target_lower']:.2f}-{row['target_upper']:.2f}%"


def _action_text(row) -> str:
    """Decision as "Hike 25bp"-style text.

    Args:
        row: Meetings row with ``action`` and ``size_bp``.

    Returns:
        Hover-ready action text.
    """
    if row["action"] == "hold":
        return "Hold"
    size = abs(int(row["size_bp"]))
    return f"{'Hike' if row['action'] == 'hike' else 'Cut'} {size}bp"


def _signed(value: float, spec: str = ".2f") -> str:
    """Format a number with an explicit sign and a true minus.

    Args:
        value: Number.
        spec: Format spec applied to the absolute value.

    Returns:
        Text such as "−0.60" or "+0.80".
    """
    text = f"{value:{spec}}"
    return ("−" + text[1:]) if text.startswith("-") else ("+" + text)


def _wrap(text: str, limit: int) -> str:
    """Break a hover-free label into two lines near a character limit.

    Args:
        text: Label text.
        limit: Target maximum characters per line.

    Returns:
        Text with a <br> at the space nearest the limit, when it is too long.
    """
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    cut = cut if cut > 0 else limit
    return text[:cut] + "<br>" + text[cut + 1 :]


# --- Ported figures -------------------------------------------------------------


def who_does_what() -> go.Figure:
    """Who-does-what schematic with a role note on hover for every box.

    Returns:
        Responsive schematic figure.
    """
    fig = go.Figure()
    for name, color in (("Federal Reserve", MAIN["BLUE"]), ("Market participants", MAIN["YELLOW"])):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(symbol="square", color=color),
                hoverinfo="skip",
            )
        )
    for x0, x1, y0, y1, main, sub, side in _BOXES:
        accent = MAIN["BLUE"] if side == "fed" else MAIN["YELLOW"]
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=PANEL,
            line_width=0, layer="below",
        )
        fig.add_shape(
            type="rect", x0=x0, y0=y1 - 0.07, x1=x1, y1=y1, fillcolor=accent,
            line_width=0, layer="below",
        )
        rate = sub if (sub and sub.endswith("%")) else None
        note = None if rate else sub
        cx, pad = (x0 + x1) / 2, 0.3
        if rate is None and note is None:
            fig.add_annotation(
                x=cx, y=(y0 + y1) / 2, text=_wrap(main, 16), showarrow=False,
                font=dict(family=FONT_BODY, color=TEXT, size=12.5),
            )
        elif rate is not None:
            fig.add_annotation(
                x=cx, y=y1 - pad, text=_wrap(main, 12), showarrow=False,
                font=dict(family=FONT_BODY, color=TEXT, size=12),
            )
            fig.add_annotation(
                x=cx, y=y0 + pad, text=f"<b>{rate}</b>", showarrow=False,
                font=dict(family=FONT_BODY, color=TEXT, size=14),
            )
            fig.add_annotation(
                x=cx, y=y0 + 0.06, text=_SET_BY[main], showarrow=False,
                font=dict(family=FONT_BODY, color=MUTED, size=8),
            )
        else:
            fig.add_annotation(
                x=cx, y=y1 - pad, text=_wrap(main, 12), showarrow=False,
                font=dict(family=FONT_BODY, color=TEXT, size=12.5),
            )
            fig.add_annotation(
                x=cx, y=y0 + pad, text=_wrap(note, 24), showarrow=False,
                font=dict(family=FONT_BODY, color=MUTED, size=12),
            )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                mode="lines",
                name=main,
                hoverinfo="text",
                hovertext=_BOX_NOTES[main] + (f" {rate}" if rate else ""),
                showlegend=False,
            )
        )
    # Schematic arrows: pixel offsets tuned at the article-column width.
    sx = (WEB_WIDTH - 2 * _PAD) / 10.0
    sy = (_H_WHO - _LEG_BAND - 8.0) / (10.0 - 1.6)
    for x0, y0, x1, y1, label, lx, ly, anchor, dash, width in _ARROWS:
        ax, ay = (x0 - x1) * sx, -(y0 - y1) * sy
        size, so = (1.3, 0) if dash else (0.9, 1)
        if dash:
            fig.add_shape(
                type="line", x0=x0, y0=y0, x1=x1, y1=y1, xref="x", yref="y",
                line=dict(color=_ARROW, width=width, dash=dash), layer="above",
            )
            norm = (ax * ax + ay * ay) ** 0.5
            ax, ay = 8.0 * ax / norm, 8.0 * ay / norm
        fig.add_annotation(
            x=x1, y=y1, ax=ax, ay=ay, xref="x", yref="y", showarrow=True,
            arrowhead=2, arrowsize=size, arrowwidth=width,
            arrowcolor=BLACK if width > 2 else _ARROW, standoff=so,
        )
        if label:
            text = label if "<br>" in label else _wrap(label, 24)
            fig.add_annotation(
                x=lx, y=ly, xanchor=anchor, text=text, showarrow=False,
                font=dict(family=FONT_BODY, size=12, color=TEXT),
            )
    for text, gx in _GROUPS:
        fig.add_annotation(
            x=gx, y=6.0, text=text, showarrow=False,
            font=dict(family=FONT_BODY, weight=300, size=13, color=TEXT),
        )
    note_text, nx, ny = _NOTE
    fig.add_annotation(
        x=nx, y=ny, text=note_text, showarrow=False,
        font=dict(family=FONT_BODY, size=9.5, color=MUTED),
    )
    _base(fig, height=_H_WHO, hovermode="closest", left=_PAD, right=_PAD,
          top=_LEG_BAND, bottom=8.0)
    fig.update_xaxes(range=[0, 10], visible=False)
    fig.update_yaxes(range=[1.6, 10.0], visible=False)
    fig.update_layout(meta=dict(minWidth=620))
    return fig


def corridor_vs_floor() -> go.Figure:
    """Corridor-versus-floor schematic as two panels, one per regime.

    Returns:
        Responsive two-panel schematic figure, scarce reserves beside ample.
    """
    supply = SCALES["GREY"][1]
    note_font = dict(family=FONT_BODY, size=11, color=TEXT)

    def line(x: list[float], y: list[float], color: str, width: float = 2.0,
             dash: str | None = None) -> go.Scatter:
        return go.Scatter(
            x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=color, width=width, dash=dash or "solid"),
        )

    def points(xs: list[float], ys: list[float], text: str) -> go.Scatter:
        return go.Scatter(
            x=xs, y=ys, mode="markers", marker=dict(color=BLACK, size=6),
            hoverinfo="text", hovertext=[text] * len(xs), showlegend=False,
        )

    def note(x, y, text, *, ax: float = 0, ay: float = 0, align: str = "center",
             arrow: bool = False) -> dict:
        return dict(
            x=x, y=y, ax=ax, ay=ay, text=text,
            showarrow=arrow, arrowhead=2, arrowsize=0.8, arrowwidth=0.9,
            arrowcolor=_ARROW, standoff=3 if arrow else 0, align=align,
            xanchor=align, font=note_font,
        )

    slope = (0.9 - 6.2) / (4.8 - 0.6)
    a_traces = [
        line([0.6, 4.8], [6.2, 0.9], MAIN["BLUE"]),
        line([0, 10], [6.6, 6.6], MAIN["GREY"], width=1.2, dash="dash"),
        line([2.0, 2.0], [0.5, 6.6], supply, width=1.4),
        line([3.2, 3.2], [0.5, 6.6], supply, width=1.4),
        points([2.0, 3.2], [6.2 + slope * 1.4, 6.2 + slope * 2.6],
               "With scarce reserves, moving their supply moves the rate"),
    ]
    b_traces = [
        line([0.6, 2.5], [6.2, 1.0], MAIN["BLUE"]),
        line([2.5, 9.5], [1.0, 1.0], MAIN["BLUE"]),
        line([0, 10], [1.2, 1.2], MAIN["GREEN"], width=1.2, dash="dash"),
        line([0, 10], [0.75, 0.75], MAIN["CYAN"], width=1.2, dash="dash"),
        line([5.0, 5.0], [0.4, 3.2], supply, width=1.4),
        line([7.0, 7.0], [0.4, 3.2], supply, width=1.4),
        points([5.0, 7.0], [1.0, 1.0],
               "With ample reserves, the rate sits on the interest rate on reserves"),
    ]
    a_anns = [
        note(3.2, 6.2 + slope * 2.6, "The Desk moves the supply<br>of reserves to "
             "set the rate", ax=58, ay=-34, arrow=True),
        note(9.7, 6.3, "Discount rate (ceiling)", align="right"),
        note(1.88, 6.25, "S", align="right"),
        note(3.32, 6.25, "S'", align="left"),
        note(4.3, 1.7, "Demand for reserves", align="left"),
        note(1.85, 6.2 + slope * 1.4, "Target", align="right"),
        note(0.12, 0.32, "Scarce reserves", align="left"),
        note(9.7, 5.0, "Reserves earned no interest<br>before 2008, so demand is "
             "steep<br>and quantity sets the rate", align="right"),
    ]
    b_anns = [
        note(6.0, 1.32, "Moving the quantity<br>no longer moves the rate", ay=-120,
             arrow=True),
        note(0.12, 1.42, "IORB", align="left"),
        note(0.12, 0.48, "ON RRP", align="left"),
        note(1.4, 5.2, "Demand for reserves", align="left"),
        note(7.6, 0.32, "Ample reserves", align="left"),
    ]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)
    for trace in a_traces:
        fig.add_trace(trace, row=1, col=1)
    for trace in b_traces:
        fig.add_trace(trace, row=1, col=2)

    def placed(anns: list[dict], xref: str, yref: str) -> list[dict]:
        return [{**ann, "xref": xref, "yref": yref} for ann in anns]

    fig.layout.annotations = tuple(  # type: ignore[assignment]
        placed(a_anns, "x", "y") + placed(b_anns, "x2", "y2")
    )
    _period_label(fig, 5.0, "Before 2008", angle=0)
    fig.add_annotation(
        x=0.5, y=1, xref="x2 domain", yref="paper", yanchor="top", yshift=-2,
        text="SINCE 2008", showarrow=False, font=_font("period"),
    )
    _base(fig, height=430, hovermode="closest", legend=False, top=30, left=42,
          right=10, bottom=46)
    fig.update_layout(meta=dict(minWidth=560))
    fig.update_xaxes(title=dict(text="Quantity of reserves", font=_font("tick"),
                                standoff=6))
    fig.update_yaxes(
        title=dict(text="Interest rate", font=_font("tick"), standoff=6),
        range=[0, 7.2], showgrid=False, ticks="", showticklabels=False,
    )
    fig.update_xaxes(range=[0, 10], ticks="", showticklabels=False)
    fig.update_yaxes(title=None, row=1, col=2)
    fig.update_yaxes(side="left", row=1, col=1)
    return fig


def administered_rates() -> go.Figure:
    """Daily administered rates inside the target band, with range buttons.

    Returns:
        Responsive time-series figure with a plotly rangeselector.
    """
    start = pd.Timestamp("2013-09-01")
    meetings = fomc.load_meetings()
    prior = meetings.loc[meetings["date"] < start].iloc[-1]
    in_win = meetings.loc[meetings["date"] >= start, ["date", "target_lower", "target_upper"]]
    end = in_win["date"].iloc[-1]
    band = pd.concat(
        [
            pd.DataFrame(
                {"date": [start], "target_lower": [prior["target_lower"]],
                 "target_upper": [prior["target_upper"]]}
            ),
            in_win,
            pd.DataFrame(
                {"date": [end], "target_lower": [in_win["target_lower"].iloc[-1]],
                 "target_upper": [in_win["target_upper"].iloc[-1]]}
            ),
        ],
        ignore_index=True,
    )
    effr = fred.fetch("EFFR").loc[start:]
    ior = transforms.splice(fred.fetch("IORB"), fred.fetch("IOER"), at="2021-07-29").loc[start:]
    onrrp = fred.fetch("RRPONTSYAWARD").loc[start:]
    # No standing-repo-rate series exists on FRED; the discount rate (FRED
    # RIFSRPF02ND, the primary credit rate) serves as the administered ceiling.
    discount = fred.fetch("RIFSRPF02ND").loc[start:]

    fig = go.Figure()
    fig.add_scatter(
        x=band["date"],
        y=band["target_upper"],
        customdata=band["target_lower"].to_numpy(),
        mode="lines",
        name="Target range",
        line=dict(color=GREY_LABEL, width=0.75),
        line_shape="hv",
        legendrank=1,
        legendgroup="band",
        hovertemplate="Target range %{customdata:.2f}–%{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=band["date"],
        y=band["target_lower"],
        mode="lines",
        line=dict(color=GREY_LABEL, width=0),
        fill="tonexty",
        fillcolor="rgba(164,189,201,0.22)",
        line_shape="hv",
        legendgroup="band",
        showlegend=False,
        hoverinfo="skip",
    )
    for y, name, color, width, dash, rank in (
        (effr, "EFFR", MAIN["BLUE"], 2.6, "solid", 2),
        (ior, "Rate on reserves", MAIN["GREEN"], 1.7, "solid", 3),
        (onrrp, "ON RRP", MAIN["CYAN"], 1.7, "dash", 4),
        (discount, "Discount rate (ceiling)", SCALES["GREY"][0], 1.2, "dot", 5),
    ):
        fig.add_scatter(
            x=y.index,
            y=y.to_numpy(),
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
            connectgaps=True,
            legendrank=rank,
            hovertemplate=f"{name} " + "%{y:.2f}%<extra></extra>",
        )
    _base(fig, height=440, top=64)
    last = max(end, effr.index.max()) + pd.Timedelta(days=5)

    def view(offset) -> dict:
        left = last - offset if offset is not None else start
        lo = onrrp.loc[left:].min()
        hi = max(band.loc[band["date"] >= left, "target_upper"].max(), discount.loc[left:].max())
        pad = 0.15 if offset is not None and offset.years == 1 else 0.3
        fmt = "%b %Y" if offset is not None and offset.years == 1 else "%Y"
        return {"xaxis.range": [str(left.date()), str(last.date())],
                "yaxis.range": [max(lo - pad, 0), hi + pad], "xaxis.tickformat": fmt}

    fig.update_xaxes(type="date")
    fig.update_yaxes(dtick=0.25)
    views = {label: view(offset) for label, offset in (
        ("1y", pd.DateOffset(years=1)), ("3y", pd.DateOffset(years=3)),
        ("5y", pd.DateOffset(years=5)), ("All", None)
    )}
    buttons = [
        dict(label=label, method="relayout",
             args=[{**v, "yaxis.dtick": 0.25 if label == "1y" else 1}])
        for label, v in views.items()
    ]
    _menu(fig, buttons, active=0)
    _stack_top(fig, height=440, top=64)
    first = views["1y"]
    fig.layout.xaxis.range = first["xaxis.range"]
    fig.layout.xaxis.tickformat = first["xaxis.tickformat"]
    fig.layout.yaxis.range = first["yaxis.range"]
    return fig


def target_rate_history() -> go.Figure:
    """Thirty years of target decisions with chair shading and zoom buttons.

    Returns:
        Responsive time-series figure with per-chair range buttons.
    """
    x_start = pd.Timestamp("1994-01-01")
    x_end = pd.Timestamp("2026-10-01")
    meetings = fomc.load_meetings()
    mid = meetings["target"].fillna((meetings["target_lower"] + meetings["target_upper"]) / 2.0)
    dates = pd.DatetimeIndex(list(meetings["date"]) + [x_end])
    mid = pd.Series(list(mid) + [mid.iloc[-1]], index=dates)
    chairs = fomc.chair_at(dates).to_numpy()
    targets = [_target_text(r) for _, r in meetings.iterrows()]
    targets.append(targets[-1])

    post = meetings.loc[meetings["date"] >= pd.Timestamp("2008-12-16"),
                        ["date", "target_lower", "target_upper"]]
    band = pd.concat(
        [
            post,
            pd.DataFrame(
                {"date": [x_end], "target_lower": [post["target_lower"].iloc[-1]],
                 "target_upper": [post["target_upper"].iloc[-1]]}
            ),
        ],
        ignore_index=True,
    )

    fig = go.Figure()
    _shade(fig, _recession_spans(), _REC_FILL)
    spans = _chair_spans(x_end)
    total = (x_end - x_start).days
    for i, (name, s, e) in enumerate(spans):
        if e < x_start:
            continue
        s = max(s, x_start)
        if i > 0:  # thin divider at each handover; chairs stay unshaded
            fig.add_shape(
                type="line", xref="x", yref="paper", x0=s, x1=s, y0=0, y1=1,
                line=dict(color=_CHAIR_RULE, width=0.72, dash="dot"), layer="below",
            )
        px = (e - s).days / total * (WEB_WIDTH - _PAD - _Y_ROOM)
        if px >= 20 or (i == len(spans) - 1 and e == x_end):
            _period_label(fig, s + (e - s) / 2, name, angle=-90 if px < 70 else 0)

    fig.add_scatter(
        x=mid.index,
        y=mid.to_numpy(),
        customdata=np.column_stack([chairs, targets]),
        mode="lines",
        name="Target midpoint",
        line=dict(color=MAIN["BLUE"], width=2.2),
        line_shape="hv",
        legendrank=1,
        hovertemplate="%{customdata[0]} chair<br>Target midpoint %{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=band["date"],
        y=band["target_upper"],
        mode="lines",
        name="Target range",
        line=dict(color=GREY_LABEL, width=0.75),
        line_shape="hv",
        legendgroup="band",
        legendrank=2,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=band["date"],
        y=band["target_lower"],
        mode="lines",
        line=dict(color=GREY_LABEL, width=0),
        fill="tonexty",
        fillcolor="rgba(164,189,201,0.25)",
        line_shape="hv",
        legendgroup="band",
        showlegend=False,
        hoverinfo="skip",
    )
    moved = meetings.loc[meetings["size_bp"] != 0]
    fig.add_scatter(
        x=moved["date"],
        y=mid.asof(pd.DatetimeIndex(moved["date"])).to_numpy(),
        customdata=np.column_stack(
            [
                [_action_text(r) for _, r in moved.iterrows()],
                [_target_text(r) for _, r in moved.iterrows()],
                fomc.chair_at(pd.DatetimeIndex(moved["date"])).to_numpy(),
            ]
        ),
        mode="markers",
        marker=dict(color=MAIN["BLUE"], size=4.5, line=dict(width=0)),
        name="Decision",
        showlegend=False,
        hovertemplate="%{x|%d %b %Y}<br>%{customdata[0]} to %{customdata[1]}"
        "<br>Chair: %{customdata[2]}<extra></extra>",
    )
    for date, text, ax, ay in (
        ("1994-02-04", "First announced change,<br>Feb 4th 1994", 70, 26),
        ("2008-12-16", "Cut to 0-0.25%", -30, -26),
        ("2015-12-16", "Liftoff", -64, -30),
        ("2020-03-15", "Back to zero", 6, -26),
        ("2026-09-16", "Warsh's first rise", -70, 45),
    ):
        fig.add_annotation(
            xref="x", yref="y", x=date, y=float(mid.asof(pd.Timestamp(date))), ax=ax,
            ay=ay, text=text, showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=0.9, arrowcolor=_ARROW, standoff=3, align="center",
            font=_font("annotation"),
        )
    _base(fig, height=470, top=64)
    fig.update_xaxes(range=[x_start, x_end], tickformat="%Y")
    fig.update_yaxes(range=[0, 7.15], tickvals=list(range(7)))
    buttons = [
        dict(label="All", method="relayout",
             args=[{"xaxis.range": [str(x_start.date()), str(x_end.date())]}])
    ]
    for name, s, e in spans:
        if e < x_start:
            continue
        buttons.append(
            dict(label=name, method="relayout",
                 args=[{"xaxis.range": [str(max(s, x_start).date()), str(e.date())]}])
        )
    _menu(fig, buttons, dropdown=True)
    _stack_top(fig, height=470, top=64)
    return fig


def specification_grid() -> go.Figure:
    """Level-model coefficients across the six-cell specification grid.

    Returns:
        Responsive two-panel dot-and-interval figure with hover CIs.
    """
    grid = level.specification_grid(
        dataset.load_frame("analysis_monthly"), dataset.build_frame("q")
    )
    wide = grid.set_index(["freq", "start"])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.10)
    for col, (coef, unit) in enumerate((("a", "Inflation"), ("b", "Jobs market")), start=1):
        for freq in ("M", "Q"):
            name, color = _FREQ[freq]
            idx = [i for i, row in enumerate(_ROWS) if row.endswith(freq)]
            est = []
            lo, hi = [], []
            for i in idx:
                key = (_FREQ_NAME[_ROWS[i][-1]], _ROWS[i][:-2])
                value = float(wide.loc[key, coef])
                half = 1.96 * float(wide.loc[key, coef + "_se"])
                est.append(value)
                lo.append(value - half)
                hi.append(value + half)
            labels = [
                f"{_FREQ_NAME[_ROWS[i][-1]].capitalize()}, {_ROWS[i][:-2]}" for i in idx
            ]
            fig.add_trace(
                go.Scatter(
                    y=idx,
                    x=est,
                    mode="markers",
                    name=name,
                    legendgroup=freq,
                    showlegend=col == 1,
                    marker=dict(color=color, size=7),
                    customdata=np.column_stack(
                        [labels, est, [_signed(v) for v in lo], [_signed(v) for v in hi]]
                    ),
                    error_x=dict(
                        type="data", symmetric=False,
                        array=[h - e for h, e in zip(hi, est, strict=True)],
                        arrayminus=[e - low for e, low in zip(est, lo, strict=True)],
                        color=color, thickness=1.4,
                    ),
                    hovertemplate="%{customdata[0]}<br>" + unit + " %{customdata[1]}"
                    "<br>95% CI %{customdata[2]} to %{customdata[3]}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.add_shape(
            type="line", xref=f"x{col if col > 1 else ''}", yref="paper", x0=0, x1=0,
            y0=0, y1=1, line=dict(color=ECON_RED, width=0.72), layer="below",
        )
    _base(fig, height=460, left=48, top=52, bottom=46)
    fig.update_layout(legend=dict(y=1.06))
    # The 1994+ monthly inflation interval (upper bound 11.2) runs past the
    # panel edge; flag it the way the production chart does.
    fig.add_annotation(
        xref="x", yref="y", x=_SPEC_GRID_XR["a"][1] - 0.05, y=_ROWS.index("1994+ M"),
        ax=-11, ay=0, text="", showarrow=True, arrowhead=2, arrowsize=0.8,
        arrowwidth=0.9, arrowcolor=_ARROW,
    )
    for col, coef in ((1, "a"), (2, "b")):
        fig.update_xaxes(
            showgrid=False, ticks="", showline=False, range=_SPEC_GRID_XR[coef],
            dtick=_SPEC_GRID_DTICK[coef], row=1, col=col,
            title=dict(text="Long-run response, points", font=_font("tick"),
                       standoff=6),
        )
    fig.update_yaxes(
        showgrid=False, range=[-0.6, len(_ROWS) - 0.4], tickvals=[0.5, 2.5, 4.5],
        ticktext=["1994+", "1983+", "1961+"], side="left", ticklabelstandoff=8,
    )
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    for xref, text in (("x domain", "Inflation"), ("x2 domain", "Jobs market")):
        fig.add_annotation(
            x=0, y=1, xref=xref, yref="y domain", xanchor="left", yanchor="bottom",
            yshift=2, text=text, showarrow=False,
            font=dict(family=FONT_BODY, size=12.5, color=TEXT),
        )
    return fig


def rolling_inflation_response() -> go.Figure:
    """Rolling long-run inflation response with medians, windows to Dec 1999.

    Returns:
        Responsive time-series figure with hover on estimates.
    """
    x_start = pd.Timestamp("1970-12-01")
    x_end = pd.Timestamp("1999-12-01")
    break_ = pd.Timestamp("1983-01-01")
    y_lo, y_hi = -3.0, 2.0
    y_pad = 0.3
    a = level.rolling(dataset.load_frame(), 120)["a"].loc[:x_end]
    pre = float(a.loc[a.index < break_].median())
    mid = float(a.loc[a.index >= break_].median())
    in_range = a.where(a.between(y_lo, y_hi))

    fig = go.Figure()
    _shade(fig, [(x_start, break_)], PANEL)
    fig.add_scatter(
        x=a.index,
        y=in_range.to_numpy(),
        mode="lines",
        name="Window estimate",
        line=dict(color=MAIN["BLUE"], width=1.8),
        hovertemplate="Response %{y:+.2f}<extra>10-year window</extra>",
    )
    for (x0, x1), med in (((x_start, break_), pre), ((break_, x_end), mid)):
        fig.add_shape(
            type="line", xref="x", yref="y", x0=x0, x1=x1, y0=med, y1=med,
            layer="above", line=dict(color=BLACK, width=0.9, dash="dash"),
        )
    fig.add_annotation(
        xref="x", yref="y", x=pd.Timestamp("1972-06-01"), y=pre, yshift=6,
        xanchor="center", yanchor="bottom", text=f"Median {_signed(pre, '.1f')}",
        showarrow=False, font=_font("annotation"),
    )
    fig.add_annotation(
        xref="x", yref="y", x=pd.Timestamp("1987-06-01"), y=mid, yshift=7,
        xanchor="center", yanchor="bottom", text=f"Median {_signed(mid, '.1f')}",
        showarrow=False, font=_font("annotation"),
    )
    _period_label(fig, pd.Timestamp("1976-07-01"), "Windows end before 1983")
    fig.add_shape(
        type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
        layer="above", line=dict(color=ECON_RED, width=1.0),
    )
    _base(fig, height=400)
    fig.update_xaxes(range=[x_start, x_end], tickformat="%Y")
    fig.update_yaxes(range=[y_lo - y_pad, y_hi + y_pad],
                     tickvals=list(range(int(y_lo), int(y_hi) + 1)))
    return fig


def decision_ablation() -> go.Figure:
    """Ablation bars with buttons switching the scoring metric.

    Returns:
        Responsive bar figure with a metric button row.
    """
    if ABLATION_CSV.exists():
        table = pd.read_csv(ABLATION_CSV)
    else:  # pragma: no cover - cache normally present
        from fedrates.figures import decision_ablation as prod

        table = prod._ablation()
    row = table.set_index("spec")
    specs = [spec for spec, _ in ABL_ROWS]
    counts = [int(row.loc[spec, "k"]) for spec in specs]
    ticktext = [
        f"{name} ({k})" if k else name
        for (_, name), k in zip(ABL_ROWS, counts, strict=True)
    ]
    n = len(specs)
    ys = [
        float(n - 1 - i - 0.22 * sum(specs[j] in _GROUP_ENDS for j in range(i)))
        for i in range(n)
    ]
    colours = [MAIN["BLUE"] if spec == ABL_FOCUS else GREY_LABEL for spec in specs]
    metric = ABL_PANELS[0][0]
    values = [float(row.loc[spec, metric]) for spec in specs]

    def baseline(value: float) -> dict:
        return dict(
            type="line", xref="x", yref="paper", x0=value, x1=value, y0=0, y1=1,
            line=dict(color=TEXT, width=0.72, dash="dash"), layer="above",
        )

    def better_arrow(lower: bool) -> dict:
        return dict(
            xref="x domain", yref="paper", x=0.08 if lower else 0.9, y=0, yshift=-16,
            ax=70 if lower else -70, ay=0, text="Better", showarrow=True, arrowhead=2,
            arrowsize=0.9, arrowwidth=0.9, arrowcolor=TEXT,
            xanchor="left" if lower else "right", font=_font("annotation"),
        )

    fig = go.Figure(
        go.Bar(
            x=values,
            y=ys,
            orientation="h",
            marker=dict(color=colours),
            customdata=np.column_stack([ticktext, values]),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    _base(
        fig, height=480, hovermode="closest", legend=False,
        left=max(_text_width(t, "tick") for t in ticktext) + 18.0, right=10, top=34,
    )
    buttons = []
    for metric, label, rng, ticks in ABL_PANELS:
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"x": [[float(row.loc[spec, metric]) for spec in specs]]},
                    {
                        "xaxis": {
                            "range": list(rng),
                            "tickvals": [float(t) for t in ticks],
                            "ticktext": [f"{t:g}" for t in ticks],
                        },
                        "shapes": [baseline(float(row.loc["expanding_prior", metric]))],
                        "annotations": [better_arrow(metric in {"rps"})],
                    },
                ],
            )
        )
    _menu(fig, buttons)
    fig.update_xaxes(
        range=list(ABL_PANELS[0][2]),
        tickvals=[float(t) for t in ABL_PANELS[0][3]],
        ticktext=[f"{t:g}" for t in ABL_PANELS[0][3]],
        showline=True, linecolor=BLACK, linewidth=0.72,
    )
    fig.update_yaxes(
        range=[min(ys) - 0.5, max(ys) + 0.5], tickvals=ys, ticktext=ticktext,
        side="left", ticklabelstandoff=6, showgrid=False,
    )
    fig.add_annotation(**better_arrow(True))
    fig.add_shape(**baseline(float(row.loc["expanding_prior", "rps"])))
    return fig


def cut25_misses() -> go.Figure:
    """Predicted probability of a 25bp cut at each 25bp-cut meeting.

    Returns:
        Responsive dot figure with per-meeting hover.
    """
    wf = pd.read_csv(CUT25_CSV, parse_dates=["date"], index_col="date")
    proba = wf[list(decision.CLASSES)]
    predicted = pd.Series(
        np.asarray(decision.CLASSES, dtype=object)[proba.to_numpy().argmax(axis=1)],
        index=wf.index,
    )
    misses = wf.loc[wf["label"] == "cut25"].copy()
    misses["predicted"] = predicted.loc[misses.index]

    labels = {
        "hold": ("Predicted hold", MUTED),
        "cut50+": ("Predicted cut 50bp+", MAIN["YELLOW"]),
        "cut25": ("Predicted cut 25bp", MAIN["GREEN"]),
    }
    fig = go.Figure()
    for key in ("hold", "cut50+", "cut25"):
        part = misses.loc[misses["predicted"] == key]
        fig.add_scatter(
            x=part.index,
            y=part["cut25"].to_numpy() * 100.0,
            mode="markers",
            name=labels[key][0],
            marker=dict(color=labels[key][1], size=8, line=dict(width=0)),
            customdata=np.column_stack(
                [part["predicted"].map(lambda k: labels[k][0]), part["cut25"] * 100.0]
            ),
            hovertemplate="%{x|%d %b %Y}<br>Predicted: %{customdata[0]}"
            "<br>P(25bp cut): %{customdata[1]:.0f}%<br>Outcome: 25bp cut<extra></extra>",
        )
    peak = float(misses["cut25"].max()) * 100.0
    fig.add_shape(
        type="line", xref="x", yref="y", x0=misses.index.min(), x1=misses.index.max(),
        y0=peak, y1=peak, line=dict(color=TEXT, width=0.72, dash="dash"), layer="above",
    )
    fig.add_annotation(
        xref="x", yref="y", x=misses.index.min(), y=peak, yshift=6, xanchor="left",
        yanchor="bottom", text=f"Peak {peak:.0f}%", showarrow=False,
        font=_font("annotation"),
    )
    _base(fig, height=400)
    fig.update_xaxes(
        tickvals=["2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01", "2024-01-01"],
        ticktext=["2008", "12", "16", "20", "24"],
    )
    fig.update_yaxes(range=[0, 34], dtick=10)
    return fig


# --- New figures -----------------------------------------------------------------


def macro_backdrop() -> go.Figure:
    """Fed funds rate, headline CPI inflation and unemployment since 1960.

    Returns:
        Responsive two-panel figure with recession shading and range buttons.
    """
    frame = dataset.load_frame()
    end = frame.index.max()
    cpi_yoy = transforms.yoy(frame["cpi"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.6, 0.4])
    for row, series in enumerate(
        (
            (
                (frame["ffr"], "Fed funds rate", MAIN["BLUE"], 2.0),
                (cpi_yoy, "CPI inflation, y/y", MAIN["CYAN"], 1.6),
            ),
            ((frame["unrate"], "Unemployment", MAIN["GREEN"], 1.6),),
        ),
        start=1,
    ):
        for rank, (y, name, color, width) in enumerate(series, start=1 if row == 1 else 3):
            fig.add_scatter(
                x=y.index,
                y=y.to_numpy(),
                mode="lines",
                name=name,
                line=dict(color=color, width=width),
                connectgaps=True,
                legendrank=rank,
                row=row,
                col=1,
                hovertemplate=f"{name} " + "%{y:.1f}%<extra></extra>",
            )
        _shade(fig, _recession_spans(), PANEL, yref=f"y{'' if row == 1 else row} domain")
    fig.add_annotation(
        x=0, y=1, xref="paper", yref="y domain", xanchor="left", yanchor="top",
        yshift=2, text="RATES AND INFLATION", showarrow=False, font=_font("period"),
    )
    fig.add_annotation(
        x=0, y=1, xref="paper", yref="y2 domain", xanchor="left", yanchor="top",
        yshift=2, text="UNEMPLOYMENT", showarrow=False, font=_font("period"),
    )
    _base(fig, height=560, top=64)
    fig.update_xaxes(range=["1960-01-01", end], tickformat="%Y")
    fig.update_yaxes(range=[0, 20], dtick=5, row=1, col=1)
    fig.update_yaxes(range=[0, 16], dtick=4, row=2, col=1)
    buttons = []
    for label, left in (
        ("All", "1960-01-01"), ("1979-87", "1979-08-01"), ("1994-", "1994-01-01"),
        ("2008-", "2008-01-01"), ("2020-", "2020-01-01"),
    ):
        right = str(end.date()) if label in {"All", "1994-", "2008-", "2020-"} else "1988-01-01"
        buttons.append(
            dict(label=label, method="relayout",
                 args=[{"xaxis.range": [left, right], "xaxis2.range": [left, right]}])
        )
    _menu(fig, buttons)
    _stack_top(fig, height=560, top=64)
    return fig


def decision_record() -> go.Figure:
    """One mark per scheduled FOMC meeting, sized by the target change.

    Returns:
        Responsive bar figure coloured by hike, hold and cut.
    """
    meetings = fomc.load_meetings(scheduled_only=True)
    hikes = meetings.loc[meetings["size_bp"] > 0]
    cuts = meetings.loc[meetings["size_bp"] < 0]
    holds = meetings.loc[meetings["size_bp"] == 0]
    width_ms = 30 * 86400 * 1000.0

    def customdata(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                [_action_text(r) for _, r in frame.iterrows()],
                [_target_text(r) for _, r in frame.iterrows()],
                fomc.chair_at(pd.DatetimeIndex(frame["date"])).to_numpy(),
            ]
        )

    hover = "%{x|%d %b %Y}<br>%{customdata[0]}<br>Target: %{customdata[1]}"
    hover += "<br>Chair: %{customdata[2]}<extra></extra>"
    fig = go.Figure()
    fig.add_bar(
        x=hikes["date"], y=hikes["size_bp"].to_numpy(), width=width_ms,
        marker_color=MAIN["RED"], name="Rise", customdata=customdata(hikes),
        hovertemplate=hover,
    )
    fig.add_bar(
        x=cuts["date"], y=cuts["size_bp"].to_numpy(), width=width_ms,
        marker_color=MAIN["BLUE"], name="Cut", customdata=customdata(cuts),
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=holds["date"],
        y=np.zeros(len(holds)),
        mode="markers",
        name="Hold",
        marker=dict(color=MUTED, symbol="line-ns", size=5, line=dict(width=1, color=MUTED)),
        customdata=customdata(holds),
        hovertemplate=hover,
    )
    fig.add_annotation(
        xref="x", yref="y", x="2008-12-16", y=-87.5, ax=12, ay=18,
        xanchor="left", text="Dec 2008: to 0-0.25%", showarrow=True,
        arrowhead=2, arrowsize=0.8, arrowwidth=0.9, arrowcolor=_ARROW,
        standoff=3, font=_font("annotation"),
    )
    _base(fig, height=330, hovermode="closest")
    fig.update_xaxes(range=["1994-01-01", "2026-10-01"], tick0="1995-01-01",
                     dtick="M60", tickformat="%Y")
    fig.update_yaxes(range=[-105, 90], dtick=25)
    fig.update_yaxes(zeroline=True, zerolinecolor=BLACK, zerolinewidth=0.72)
    fig.update_xaxes(showline=False, ticks="")
    return fig


def rules_vs_actual() -> go.Figure:
    """The Taylor (1993) rule against the actual fed funds rate.

    Returns:
        Responsive time-series figure with a top legend.
    """
    frame = dataset.load_frame()
    fig = go.Figure()
    actual = frame["ffr"]
    fig.add_scatter(
        x=actual.index,
        y=actual.to_numpy(),
        mode="lines",
        name="Fed funds (actual)",
        line=dict(color=BLACK, width=2.2),
        legendrank=1,
        hovertemplate="Fed funds (actual) %{y:.2f}%<extra></extra>",
    )
    rule = frame["rule_taylor93"]
    fig.add_scatter(
        x=rule.index,
        y=rule.to_numpy(),
        mode="lines",
        name="Taylor (1993) rule",
        line=dict(color=MAIN["BLUE"], width=1.6),
        legendrank=2,
        hovertemplate="Taylor (1993) rule %{y:.2f}%<extra></extra>",
    )
    _base(fig, height=430, top=30)
    fig.update_xaxes(range=["1961-01-01", frame.index.max()], tickformat="%Y")
    fig.update_yaxes(range=[-0.4, 16], dtick=4)
    return fig


FIGURES: dict[str, Callable[[], go.Figure]] = {
    "who_does_what": who_does_what,
    "corridor_vs_floor": corridor_vs_floor,
    "administered_rates": administered_rates,
    "target_rate_history": target_rate_history,
    "specification_grid": specification_grid,
    "rolling_inflation_response": rolling_inflation_response,
    "decision_ablation": decision_ablation,
    "cut25_misses": cut25_misses,
    "macro_backdrop": macro_backdrop,
    "decision_record": decision_record,
    "rules_vs_actual": rules_vs_actual,
}


def build_all() -> dict[str, go.Figure]:
    """Build every web figure, keyed by figure id.

    Returns:
        Dict of figure id to plotly figure, in contract order.
    """
    register()
    return {fid: build() for fid, build in FIGURES.items()}


def _main() -> None:
    """Write inspection copies: HTML with a CDN script and a 640px-wide PNG."""
    register()
    OUT.mkdir(parents=True, exist_ok=True)
    for fid, build in FIGURES.items():
        fig = build()
        fig.write_html(OUT / f"{fid}.html", include_plotlyjs="cdn", full_html=True)
        height = int(fig.layout.height)
        fig.update_layout(width=WEB_WIDTH, autosize=False)
        fig.write_image(OUT / f"{fid}.png", width=WEB_WIDTH, height=height, scale=2)
        print(f"wrote {OUT / fid}.html and .png")


if __name__ == "__main__":
    _main()
