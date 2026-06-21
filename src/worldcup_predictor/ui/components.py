from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _full_width_kwargs() -> dict[str, object]:
    """Fill the container width in a way that works across Streamlit versions.

    Streamlit 1.54 replaced ``use_container_width=True`` with ``width="stretch"``.
    Passing the string form to an older build raises a TypeError, so pick the
    argument that matches the installed version.
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except ValueError:
        return {"width": "stretch"}
    return {"width": "stretch"} if (major, minor) >= (1, 54) else {"use_container_width": True}


FULL_WIDTH = _full_width_kwargs()

COLOR = {
    "ink": "#050b14",
    "navy": "#071321",
    "panel": "#0d1c2d",
    "line": "#1c334a",
    "text": "#f5f8fc",
    "muted": "#8fa5bb",
    "cyan": "#45e6c2",
    "blue": "#5b8ff9",
    "gold": "#f5c85b",
    "red": "#ef6a72",
}


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>""",
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, height: int = 450) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#dce7f2"},
        height=height,
        margin=dict(l=20, r=20, t=60, b=25),
        legend_title_text="",
        hoverlabel={"bgcolor": "#0d1c2d", "font_color": "#f5f8fc"},
    )
    fig.update_xaxes(gridcolor="#1c334a", zerolinecolor="#1c334a")
    fig.update_yaxes(gridcolor="#1c334a", zerolinecolor="#1c334a")
    return fig


def probability_label(
    probabilities: np.ndarray, home: str, away: str
) -> str:
    return (f"{away} win", "Draw", f"{home} win")[
        int(np.argmax(probabilities))
    ]
