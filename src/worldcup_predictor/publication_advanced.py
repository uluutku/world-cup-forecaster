from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import VISUAL_DIR

NAVY = "#07111f"
GRID = "#20334c"
TEXT = "#f4f7fb"
MUTED = "#8da3ba"
CYAN = "#45e6c2"
BLUE = "#5b8ff9"
GOLD = "#f5c85b"
RED = "#ef6a72"


def _setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": NAVY,
            "axes.facecolor": NAVY,
            "savefig.facecolor": NAVY,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )


def _save(fig: plt.Figure, filename: str) -> Path:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUAL_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return path


def _brand(fig: plt.Figure, section: str) -> None:
    fig.text(0.025, 0.982, "WORLD CUP FORECASTER", color=CYAN, fontsize=9, weight="bold", va="top")
    fig.text(0.975, 0.982, section.upper(), color=MUTED, fontsize=8, ha="right", va="top")

def advanced_research_chart(
    matrix: pd.DataFrame,
    architectures: pd.DataFrame,
) -> Path:
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.subplots_adjust(top=0.78, wspace=0.46)
    _brand(fig, "Advanced research laboratory")
    fig.suptitle(
        "Complexity had to earn its place",
        x=0.025,
        y=0.93,
        ha="left",
        fontsize=28,
        weight="bold",
    )
    fig.text(
        0.025,
        0.865,
        "Chronological World Cup tests · lower log loss is better · oracle and lineup-only signals are explicitly separated",
        color=MUTED,
        fontsize=11,
    )

    ax = axes[0]
    architecture = (
        architectures.groupby("architecture", as_index=False)
        .agg(log_loss=("log_loss", "mean"), accuracy=("accuracy", "mean"))
        .sort_values("log_loss", ascending=False)
    )
    colors = [
        CYAN if name == "Calibrated hybrid" else BLUE
        for name in architecture["architecture"]
    ]
    bars = ax.barh(architecture["architecture"], architecture["log_loss"], color=colors)
    ax.set_title("Architecture benchmark", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Mean log loss · 5 folds")
    ax.set_xlim(max(0.90, architecture["log_loss"].min() - 0.025), architecture["log_loss"].max() + 0.025)
    ax.grid(axis="x", alpha=0.4)
    for bar, value, accuracy in zip(
        bars, architecture["log_loss"], architecture["accuracy"]
    ):
        ax.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} · {accuracy:.1%}",
            va="center",
            color=TEXT,
            fontsize=9,
        )

    ax = axes[1]
    candidates = matrix.loc[
        ~matrix["candidate"].isin(
            ["Baseline scores + state", "Calibrated hybrid"]
        )
    ].copy()
    candidates = candidates.sort_values("log_loss_basis_points", ascending=False)
    palette = {
        "Promotion candidate": CYAN,
        "Coverage-limited": GOLD,
        "Oracle only": BLUE,
        "Rejected": RED,
    }
    bars = ax.barh(
        candidates["candidate"],
        candidates["log_loss_basis_points"],
        color=[palette.get(value, MUTED) for value in candidates["decision"]],
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title("Data-family impact", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Log-loss change · basis points")
    ax.grid(axis="x", alpha=0.4)
    for bar, value in zip(bars, candidates["log_loss_basis_points"]):
        ax.text(
            value + (2 if value >= 0 else -2),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color=TEXT,
            fontsize=8,
        )

    ax = axes[2]
    for name, frame in architectures.groupby("architecture"):
        ax.plot(
            frame["edition"],
            frame["log_loss"],
            marker="o",
            lw=2.6 if name == "Calibrated hybrid" else 1.8,
            color=CYAN if name == "Calibrated hybrid" else BLUE if "XGBoost" in name else GOLD,
            alpha=1 if name == "Calibrated hybrid" else 0.78,
            label=name,
        )
    ax.set_title("Temporal stability", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("World Cup edition")
    ax.set_ylabel("Log loss")
    ax.set_xticks(sorted(architectures["edition"].unique()))
    ax.grid(alpha=0.4)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.025,
        0.025,
        "Result: RTX 4070 Ti training works, but neither CUDA candidate beats the calibrated football-specific hybrid. "
        "StatsBomb event history is promising on 2022; weather reanalysis and travel are rejected.",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "advanced-research.png")


def squad_intelligence_chart(squads: pd.DataFrame) -> Path:
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.subplots_adjust(top=0.78, wspace=0.42)
    _brand(fig, "Official 2026 squads")
    fig.suptitle(
        "The tournament through 1,248 player records",
        x=0.025,
        y=0.93,
        ha="left",
        fontsize=28,
        weight="bold",
    )
    fig.text(
        0.025,
        0.865,
        "Official caps, goals and height joined to age, club geography, positional depth and concentration",
        color=MUTED,
        fontsize=11,
    )

    panels = [
        ("total_caps", "International experience", "Total squad caps", 12),
        (
            "total_international_goals",
            "Proven scoring",
            "Squad international goals",
            12,
        ),
        (
            "top_five_league_share",
            "Elite-league exposure",
            "Share in ENG / ESP / GER / ITA / FRA",
            12,
        ),
    ]
    for ax, (column, title, label, count) in zip(axes, panels):
        view = squads.nlargest(count, column).sort_values(column)
        colors = [GOLD if code == "TUR" else CYAN for code in view["fifa_code"]]
        bars = ax.barh(view["team"], view[column], color=colors, alpha=0.9)
        ax.set_title(title, loc="left", fontsize=15, weight="bold")
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.4)
        if "share" in column:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{value:.0%}")
            )
        for bar, value in zip(bars, view[column]):
            text = f"{value:.0%}" if "share" in column else f"{value:,.0f}"
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {text}",
                va="center",
                color=TEXT,
                fontsize=8,
            )
    fig.text(
        0.025,
        0.025,
        "Türkiye is highlighted in gold when it enters a top-12 panel. Squad features remain a current intelligence layer "
        "until equivalent historical snapshots support a fair walk-forward test.",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "squad-intelligence.png")



