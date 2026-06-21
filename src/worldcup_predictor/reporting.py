from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, RegularPolygon

from .config import VISUAL_DIR
from .publication_advanced import (
    advanced_research_chart,
    squad_intelligence_chart,
)

NAVY = "#07111f"
PANEL = "#0e1c2f"
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


def hero_background() -> Path:
    _setup()
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    x = np.linspace(0, 1, 1200)
    y = np.linspace(0, 1, 675)
    xx, yy = np.meshgrid(x, y)
    glow = np.exp(-((xx - 0.78) ** 2 / 0.035 + (yy - 0.52) ** 2 / 0.11))
    secondary = np.exp(-((xx - 0.93) ** 2 / 0.08 + (yy - 0.18) ** 2 / 0.09))
    image = np.zeros((len(y), len(x), 3))
    base = np.array([5, 11, 20]) / 255
    image[:] = base
    image += glow[..., None] * np.array([0.015, 0.10, 0.12])
    image += secondary[..., None] * np.array([0.02, 0.04, 0.10])
    ax.imshow(np.clip(image, 0, 1), extent=[0, 16, 0, 9], origin="lower")

    for radius, alpha, color in (
        (2.8, 0.09, BLUE),
        (2.2, 0.12, CYAN),
        (1.72, 0.16, CYAN),
    ):
        ax.add_patch(
            Arc(
                (12.2, 4.55),
                radius * 2.5,
                radius,
                angle=12,
                theta1=15,
                theta2=335,
                color=color,
                lw=1.3,
                alpha=alpha,
            )
        )
    for _ in range(175):
        px = rng.normal(12.2, 2.1)
        py = rng.normal(4.5, 1.65)
        if px < 7.2:
            continue
        size = rng.uniform(2, 18)
        color = CYAN if rng.random() > 0.2 else GOLD
        ax.scatter(px, py, s=size, color=color, alpha=rng.uniform(0.08, 0.42), linewidths=0)

    ball = Circle((12.35, 4.55), 1.35, facecolor="#0b1d2c", edgecolor=CYAN, lw=1.8, alpha=0.97)
    ax.add_patch(ball)
    ax.add_patch(Circle((12.0, 4.92), 1.05, facecolor="#173348", edgecolor="none", alpha=0.56))
    for angle in np.linspace(0, 2 * np.pi, 7, endpoint=False):
        cx = 12.35 + 0.64 * np.cos(angle)
        cy = 4.55 + 0.64 * np.sin(angle)
        ax.add_patch(
            RegularPolygon(
                (cx, cy),
                numVertices=6,
                radius=0.31,
                orientation=angle / 2,
                facecolor="#102b3d",
                edgecolor=BLUE,
                lw=0.7,
                alpha=0.72,
            )
        )
    ax.add_patch(
        RegularPolygon(
            (12.35, 4.55),
            numVertices=5,
            radius=0.38,
            orientation=0.3,
            facecolor="#07131f",
            edgecolor=CYAN,
            lw=0.9,
            alpha=0.88,
        )
    )
    ax.add_patch(Arc((11.92, 4.98), 1.65, 1.65, theta1=80, theta2=210, color="#d7fff7", lw=2.2, alpha=0.25))

    for start_y, color, alpha in ((1.3, BLUE, 0.18), (2.5, CYAN, 0.25), (6.8, GOLD, 0.13)):
        points_x = np.linspace(7.4, 15.8, 70)
        curve = start_y + 0.85 * np.sin(points_x * 0.9 + start_y)
        ax.plot(points_x, curve, color=color, alpha=alpha, lw=1)
        indices = rng.choice(len(points_x), 10, replace=False)
        ax.scatter(points_x[indices], curve[indices], color=color, alpha=alpha * 1.8, s=8)
    return _save(fig, "hero-data-field.png")


def model_evidence_chart(
    backtests: pd.DataFrame,
    benchmarks: pd.DataFrame,
    reliability: pd.DataFrame,
    ablation: pd.DataFrame,
    research_note: str | None = None,
) -> Path:
    _setup()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.36, wspace=0.27, top=0.84)
    _brand(fig, "Model evidence")
    fig.suptitle(
        "Evidence before confidence",
        x=0.025,
        y=0.94,
        ha="left",
        fontsize=26,
        weight="bold",
    )
    fig.text(
        0.025,
        0.875,
        "Five tournament cutoffs · every prediction made with information available before kickoff",
        color=MUTED,
        fontsize=11,
    )

    ax = axes[0, 0]
    ax.plot(backtests["edition"], backtests["baseline_log_loss"], "--", color=MUTED, lw=2, label="Historical prior")
    ax.plot(backtests["edition"], backtests["log_loss"], "-o", color=CYAN, lw=3, ms=7, label="Calibrated ensemble")
    ax.fill_between(
        backtests["edition"],
        backtests["log_loss"],
        backtests["baseline_log_loss"],
        where=backtests["log_loss"] <= backtests["baseline_log_loss"],
        color=CYAN,
        alpha=0.09,
    )
    ax.set_title("Walk-forward log loss", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Lower is better")
    ax.set_xticks(backtests["edition"])
    ax.grid(alpha=0.55)
    ax.legend(frameon=False, labelcolor=TEXT)

    ax = axes[0, 1]
    ladder = (
        benchmarks.groupby("model", as_index=False)["log_loss"]
        .mean()
        .sort_values("log_loss", ascending=True)
    )
    colors = [CYAN if name == "Calibrated ensemble" else BLUE for name in ladder["model"]]
    bars = ax.barh(ladder["model"], ladder["log_loss"], color=colors, alpha=0.92)
    ax.invert_yaxis()
    ax.set_title("Benchmark ladder", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Mean log loss across World Cups")
    ax.grid(axis="x", alpha=0.5)
    for bar, value in zip(bars, ladder["log_loss"]):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color=TEXT, fontsize=9)

    ax = axes[1, 0]
    for outcome, group in reliability.groupby("outcome"):
        color = {"Home win": CYAN, "Draw": GOLD, "Away win": RED}[outcome]
        ax.scatter(group["predicted"], group["observed"], s=np.maximum(group["count"], 4) * 4, color=color, alpha=0.78, label=outcome, edgecolor=NAVY)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.5)
    ax.set(xlim=(0, 0.9), ylim=(0, 0.9), xlabel="Predicted probability", ylabel="Observed frequency")
    ax.set_title("Reliability surface", loc="left", fontsize=15, weight="bold")
    ax.grid(alpha=0.5)
    ax.legend(frameon=False, labelcolor=TEXT, ncol=3, fontsize=9)

    ax = axes[1, 1]
    view = ablation.sort_values("log_loss_delta", ascending=False)
    colors = [CYAN if value <= 0 else RED for value in view["log_loss_delta"]]
    bars = ax.barh(view["configuration"], view["log_loss_delta"], color=colors, alpha=0.9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title("2022 feature ablation", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Log-loss change vs full system")
    ax.grid(axis="x", alpha=0.5)
    for bar, value in zip(bars, view["log_loss_delta"]):
        ax.text(value + (0.002 if value >= 0 else -0.002), bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", color=TEXT, fontsize=9)
    if research_note:
        fig.text(0.025, 0.015, research_note, color=MUTED, fontsize=9)
    return _save(fig, "model-evidence.png")


def tournament_forecast_chart(
    simulation: pd.DataFrame,
    research_note: str | None = None,
) -> Path:
    _setup()
    top = simulation.head(16).sort_values("champion")
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(top=0.84)
    _brand(fig, "2026 tournament forecast")
    fig.suptitle("The title race is a distribution, not a ranking", x=0.03, y=0.94, ha="left", fontsize=25, weight="bold")
    fig.text(0.03, 0.875, "Monte Carlo paths through the live group state and expanded 48-team bracket", color=MUTED)
    cmap = LinearSegmentedColormap.from_list("wci", [BLUE, CYAN])
    colors = cmap(np.linspace(0.1, 1, len(top)))
    bars = ax.barh(top["team"], top["champion"], color=colors, height=0.66)
    ax.set_xlim(0, max(top["champion"]) * 1.2)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Champion probability")
    ax.grid(axis="x", alpha=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value, group in zip(bars, top["champion"], top["group"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.1%}  ·  Group {group}", va="center", color=TEXT, fontsize=10, weight="bold")
    if research_note:
        fig.text(0.03, 0.025, research_note, color=MUTED, fontsize=9)
    return _save(fig, "tournament-forecast.png")


def team_power_map(
    ratings: pd.DataFrame,
    tournament_teams: list[str],
    research_note: str | None = None,
) -> Path:
    _setup()
    frame = ratings.loc[ratings["team"].isin(tournament_teams)].copy()
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(top=0.84)
    _brand(fig, "Team intelligence")
    fig.suptitle("Team power map", x=0.03, y=0.94, ha="left", fontsize=25, weight="bold")
    fig.text(0.03, 0.875, "Recency-weighted attack and defence, sized by rating certainty", color=MUTED)
    certainty = 1 / np.clip(frame["rating_sigma"], 1, None)
    sizes = 180 + 2400 * (certainty - certainty.min()) / max(certainty.max() - certainty.min(), 1e-6)
    scatter = ax.scatter(
        frame["attack"],
        frame["defence"],
        c=frame["elo"],
        s=sizes,
        cmap=LinearSegmentedColormap.from_list("power", [RED, GOLD, CYAN]),
        alpha=0.83,
        edgecolors=NAVY,
        linewidths=1.5,
    )
    leaders = frame.nlargest(12, "elo")
    offsets = {
        "Brazil": (28, -4),
        "Portugal": (28, 16),
        "Netherlands": (-38, 2),
        "Colombia": (-12, -22),
        "Spain": (0, -6),
        "France": (0, -7),
    }
    for row in leaders.itertuples():
        offset = offsets.get(row.team, (0, -6))
        text = ax.annotate(
            row.team,
            (row.attack, row.defence),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha="center",
            color=TEXT,
            weight="bold",
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground=NAVY)])
    ax.invert_yaxis()
    ax.set_xlabel("Attacking output →")
    ax.set_ylabel("← Fewer goals conceded")
    ax.grid(alpha=0.45)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("Dynamic power rating", color=MUTED)
    if research_note:
        fig.text(0.03, 0.025, research_note, color=MUTED, fontsize=9)
    return _save(fig, "team-power-map.png")


def forecast_social_card(
    simulation: pd.DataFrame,
    backtests: pd.DataFrame,
    fixtures: pd.DataFrame,
    research_note: str | None = None,
) -> Path:
    _setup()
    fig = plt.figure(figsize=(16, 9))
    _brand(fig, "Forecast snapshot")
    grid = fig.add_gridspec(12, 24)
    ax_title = fig.add_subplot(grid[1:4, :])
    ax_title.axis("off")
    favorite = simulation.iloc[0]
    mean_loss = backtests["log_loss"].mean()
    baseline = backtests["baseline_log_loss"].mean()
    ax_title.text(0, 0.82, "WORLD CUP", fontsize=42, weight="bold", color=TEXT)
    ax_title.text(0, 0.33, "FORECASTER", fontsize=42, weight="bold", color=CYAN)
    ax_title.text(0.57, 0.72, "CURRENT TITLE FAVORITE", color=MUTED, fontsize=10, weight="bold")
    ax_title.text(0.57, 0.31, favorite["team"], color=TEXT, fontsize=32, weight="bold")
    ax_title.text(0.91, 0.31, f"{favorite['champion']:.1%}", color=CYAN, fontsize=32, weight="bold", ha="right")

    ax = fig.add_subplot(grid[5:11, :14])
    top = simulation.head(8).sort_values("champion")
    bars = ax.barh(top["team"], top["champion"], color=[BLUE] * 7 + [CYAN])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("TITLE PROBABILITY", loc="left", fontsize=10, color=MUTED, weight="bold")
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, top["champion"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color=TEXT, weight="bold")

    ax_metrics = fig.add_subplot(grid[5:11, 15:])
    ax_metrics.axis("off")
    uncertain = fixtures.sort_values("entropy", ascending=False).iloc[0]
    metrics = [
        ("BACKTEST LOG LOSS", f"{mean_loss:.3f}", f"{(1 - mean_loss / baseline):.1%} skill vs prior"),
        ("OUTCOME ACCURACY", f"{backtests['accuracy'].mean():.1%}", "five unseen World Cups"),
        ("MOST UNCERTAIN FIXTURE", f"{uncertain.home_team} · {uncertain.away_team}", f"entropy {uncertain.entropy:.2f}"),
    ]
    y = 0.90
    for label, value, note in metrics:
        ax_metrics.text(0, y, label, color=MUTED, fontsize=9, weight="bold")
        ax_metrics.text(0, y - 0.13, value, color=TEXT, fontsize=20, weight="bold")
        ax_metrics.text(0, y - 0.23, note, color=CYAN, fontsize=9)
        y -= 0.33
    fig.text(
        0.025,
        0.025,
        research_note
        or "No future data · calibrated · uncertainty-aware · reproducible",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "forecast-social-card.png")


def turkey_focus_chart(
    schedule: pd.DataFrame,
    scenarios: pd.DataFrame,
    simulation: pd.DataFrame,
    squad: pd.Series | None = None,
    players: pd.DataFrame | None = None,
    research_note: str | None = None,
) -> Path:
    _setup()
    fig = plt.figure(figsize=(18, 10))
    _brand(fig, "Türkiye focus")
    grid = fig.add_gridspec(14, 30)
    ax_title = fig.add_subplot(grid[1:4, :])
    ax_title.axis("off")
    turkey = simulation.loc[simulation["team"].eq("Turkey")].iloc[0]
    ax_title.text(0, 0.78, "TÜRKİYE", fontsize=44, weight="bold", color=TEXT)
    ax_title.text(
        0,
        0.25,
        "LIVE FORECAST × OFFICIAL SQUAD INTELLIGENCE",
        fontsize=22,
        weight="bold",
        color=CYAN,
    )
    ax_title.text(
        0.64,
        0.68,
        "CURRENT R32 PROBABILITY",
        color=MUTED,
        fontsize=10,
        weight="bold",
    )
    ax_title.text(
        0.64,
        0.22,
        f"{turkey['round_32']:.1%}",
        color=TEXT,
        fontsize=34,
        weight="bold",
    )
    if squad is not None:
        ax_title.text(
            0.86,
            0.68,
            "OFFICIAL SQUAD",
            color=MUTED,
            fontsize=10,
            weight="bold",
        )
        ax_title.text(
            0.86,
            0.22,
            f"{int(squad['players'])} players · {int(squad['total_caps']):,} caps",
            color=GOLD,
            fontsize=17,
            weight="bold",
        )

    ax_schedule = fig.add_subplot(grid[5:13, :9])
    ax_schedule.axis("off")
    ax_schedule.text(
        0, 1.02, "GROUP D MATCHES", color=MUTED, fontsize=10, weight="bold"
    )
    y = 0.84
    for match in schedule.sort_values("date").itertuples():
        home = "Türkiye" if match.home_team == "Turkey" else match.home_team
        away = "Türkiye" if match.away_team == "Turkey" else match.away_team
        if match.is_completed:
            score = f"{int(match.home_score)} – {int(match.away_score)}"
            verdict = "FINAL"
            headline = f"{home}  {score}  {away}"
            color = RED if (
                (match.home_team == "Turkey" and match.home_score < match.away_score)
                or (match.away_team == "Turkey" and match.away_score < match.home_score)
            ) else CYAN
        else:
            headline = f"{home} vs {away}"
            verdict = (
                f"HOME {match.p_home:.0%}  ·  DRAW {match.p_draw:.0%}  ·  "
                f"AWAY {match.p_away:.0%}"
            )
            color = GOLD
        ax_schedule.text(0, y, pd.Timestamp(match.date).strftime("%d %b").upper(), color=MUTED, fontsize=9)
        ax_schedule.text(
            0.15,
            y,
            headline,
            color=TEXT,
            fontsize=14,
            weight="bold",
        )
        ax_schedule.text(
            0.15,
            y - 0.09,
            verdict,
            color=color,
            fontsize=8,
            weight="bold",
        )
        y -= 0.29

    ax_squad = fig.add_subplot(grid[5:13, 10:20])
    if squad is not None and players is not None:
        metrics = [
            ("Caps", float(squad["total_caps"]), 1400),
            ("Goals", float(squad["total_international_goals"]), 240),
            ("Top-five", float(squad["top_five_league_share"]) * 100, 100),
            ("Youth", float(squad["under_23_share"]) * 100, 35),
            ("Domestic", float(squad["domestic_club_share"]) * 100, 100),
        ]
        values = [min(value / ceiling, 1) for _, value, ceiling in metrics]
        labels = [label for label, _, _ in metrics]
        bars = ax_squad.barh(
            labels[::-1],
            values[::-1],
            color=[GOLD, CYAN, BLUE, CYAN, GOLD],
            alpha=0.9,
        )
        ax_squad.set_xlim(0, 1)
        ax_squad.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax_squad.set_title(
            "OFFICIAL SQUAD PROFILE",
            loc="left",
            fontsize=10,
            color=MUTED,
            weight="bold",
        )
        ax_squad.grid(axis="x", alpha=0.4)
        ax_squad.spines[["top", "right", "left"]].set_visible(False)
        raw_labels = [
            f"{value:,.0f}" if label in {"Caps", "Goals"} else f"{value:.0f}%"
            for label, value, _ in metrics
        ][::-1]
        for bar, label in zip(bars, raw_labels):
            ax_squad.text(
                bar.get_width() + 0.025,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                color=TEXT,
                fontsize=9,
                weight="bold",
            )
        leading = (
            players.nlargest(3, "international_goals")[
                ["player", "international_goals"]
            ]
            .assign(
                label=lambda frame: frame["player"]
                + " · "
                + frame["international_goals"].fillna(0).astype(int).astype(str)
            )["label"]
            .tolist()
        )
        ax_squad.text(
            0,
            -0.17,
            "Leading international scorers: " + "  |  ".join(leading),
            transform=ax_squad.transAxes,
            color=MUTED,
            fontsize=8,
        )
    else:
        ax_squad.axis("off")

    ax = fig.add_subplot(grid[5:13, 21:])
    view = scenarios.sort_values("round_32")
    bars = ax.barh(
        view["scenario"],
        view["round_32"],
        color=[RED if value == 0 else CYAN for value in view["round_32"]],
        alpha=0.92,
    )
    ax.set_title("ROUND-OF-32 CHANCE BY USA MATCH RESULT", loc="left", fontsize=10, color=MUTED, weight="bold")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, view["round_32"]):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            color=TEXT,
            weight="bold",
        )
    fig.text(
        0.025,
        0.025,
        research_note
        or "Production probabilities remain baseline v2 · squad intelligence is descriptive · staged candidates are not silently blended",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "turkiye-focus.png")


def turkey_blind_chart(
    blind: pd.DataFrame,
    research_note: str | None = None,
) -> Path:
    _setup()
    fig = plt.figure(figsize=(16, 9))
    _brand(fig, "Blind forecast audit")
    grid = fig.add_gridspec(12, 24)
    ax_title = fig.add_subplot(grid[1:4, :])
    ax_title.axis("off")
    ax_title.text(
        0,
        0.78,
        "WHAT THE MODEL SAID BEFORE KICKOFF",
        fontsize=34,
        weight="bold",
        color=TEXT,
    )
    ax_title.text(
        0,
        0.28,
        "Frozen on 12 June 2026 · zero Türkiye World Cup results observed",
        fontsize=15,
        color=CYAN,
    )

    ax = fig.add_subplot(grid[5:11, :15])
    labels = blind["opponent"].tolist()
    y = np.arange(len(labels))
    team_win = blind["p_team_win"].to_numpy()
    draw = blind["p_draw"].to_numpy()
    opponent = blind["p_opponent_win"].to_numpy()
    ax.barh(y, team_win, color=CYAN, label="Türkiye win")
    ax.barh(y, draw, left=team_win, color=GOLD, label="Draw")
    ax.barh(y, opponent, left=team_win + draw, color=RED, label="Opponent win")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("PRE-TOURNAMENT OUTCOME DISTRIBUTION", loc="left", fontsize=10, color=MUTED, weight="bold")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(frameon=False, labelcolor=TEXT, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    for index, values in enumerate(zip(team_win, draw, opponent)):
        cumulative = 0
        for value in values:
            ax.text(cumulative + value / 2, index, f"{value:.0%}", ha="center", va="center", color=NAVY, fontsize=10, weight="bold")
            cumulative += value

    audit = fig.add_subplot(grid[5:11, 16:])
    audit.axis("off")
    audit.text(0, 1.02, "BLIND PICKS VS REALITY", color=MUTED, fontsize=10, weight="bold")
    y_position = 0.84
    for match in blind.itertuples():
        actual = match.actual_result if isinstance(match.actual_result, str) else "Not played"
        correctness = (
            "PENDING"
            if pd.isna(match.correct_pick)
            else "CORRECT"
            if bool(match.correct_pick)
            else "WRONG"
        )
        color = CYAN if correctness == "CORRECT" else RED if correctness == "WRONG" else GOLD
        audit.text(0, y_position, match.opponent.upper(), color=MUTED, fontsize=9, weight="bold")
        audit.text(0, y_position - 0.10, str(match.team_view_pick).replace("Turkey", "Türkiye"), color=TEXT, fontsize=15, weight="bold")
        audit.text(0, y_position - 0.20, f"Actual: {str(actual).replace('Turkey', 'Türkiye')}  ·  {correctness}", color=color, fontsize=9, weight="bold")
        y_position -= 0.31
    fig.text(
        0.025,
        0.025,
        research_note
        or "The two completed outcome picks were wrong. No retrospective fitting or hidden-score state updates.",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "turkiye-blind-forecast.png")


def blind_tournament_chart(
    forecasts: pd.DataFrame,
    leaderboard: pd.DataFrame,
    research_note: str | None = None,
) -> Path:
    _setup()
    completed = forecasts.loc[forecasts["is_completed"]].copy()
    accuracy = completed["correct_pick"].mean()
    log_loss = -np.log(
        np.clip(completed["actual_outcome_probability"], 1e-12, 1)
    ).mean()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(top=0.78, wspace=0.30)
    _brand(fig, "Blind tournament audit")
    fig.suptitle(
        "72 predictions frozen before the opening match",
        x=0.025,
        y=0.93,
        ha="left",
        fontsize=27,
        weight="bold",
    )
    fig.text(
        0.025,
        0.86,
        f"{len(completed)} scored · {accuracy:.1%} accuracy · {log_loss:.3f} log loss · zero tournament-state updates",
        color=CYAN,
        fontsize=12,
    )
    ax = axes[0]
    completed["confidence"] = completed[["p_home", "p_draw", "p_away"]].max(
        axis=1
    )
    colors = [CYAN if value else RED for value in completed["correct_pick"]]
    ax.scatter(
        completed["confidence"],
        completed["actual_outcome_probability"],
        c=colors,
        s=80,
        alpha=0.78,
        edgecolor=NAVY,
    )
    ax.plot([0, 1], [0, 1], "--", color=MUTED, alpha=0.55)
    ax.set(
        xlim=(0.3, 0.9),
        ylim=(0, 0.9),
        xlabel="Highest predicted probability",
        ylabel="Probability assigned to actual result",
    )
    ax.set_title("Confidence versus reality", loc="left", fontsize=15, weight="bold")
    ax.grid(alpha=0.45)

    ax = axes[1]
    team_view = leaderboard.loc[leaderboard["matches"] >= 2].copy()
    team_view = team_view.sort_values(["accuracy", "log_loss"]).tail(16)
    colors = [
        CYAN if value >= 0.5 else RED for value in team_view["accuracy"]
    ]
    bars = ax.barh(team_view["team"], team_view["log_loss"], color=colors, alpha=0.88)
    ax.set_title("Team audit · minimum two results", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Blind log loss · lower is better")
    ax.grid(axis="x", alpha=0.45)
    for bar, accuracy_value in zip(bars, team_view["accuracy"]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy_value:.0%}",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    if research_note:
        fig.text(0.025, 0.025, research_note, color=MUTED, fontsize=9)
    return _save(fig, "blind-tournament-audit.png")


def model_evolution_chart(
    enrichment: pd.DataFrame,
    coverage: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
) -> Path:
    _setup()
    summary = (
        enrichment.groupby("configuration", as_index=False)
        .agg(
            log_loss=("log_loss", "mean"),
            accuracy=("accuracy", "mean"),
            log_loss_delta=("log_loss_delta", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
        )
        .sort_values("log_loss")
    )
    summary["basis_points"] = summary["log_loss_delta"] * 10_000
    summary["source"] = "Open enrichment"
    if advanced is not None and not advanced.empty:
        staged = advanced.loc[
            ~advanced["candidate"].isin(
                ["Baseline scores + state", "Calibrated hybrid"]
            ),
            [
                "candidate",
                "log_loss",
                "accuracy",
                "log_loss_delta",
                "accuracy_delta",
                "log_loss_basis_points",
                "stage",
            ],
        ].rename(
            columns={
                "candidate": "configuration",
                "log_loss_basis_points": "basis_points",
                "stage": "source",
            }
        )
        original_focus = summary.loc[
            summary["configuration"].isin(
                [
                    "Baseline scores + state",
                    "+ shootout dynamics",
                    "+ all open enrichment",
                    "+ World Cup squad composition",
                ]
            )
        ]
        summary = pd.concat([original_focus, staged], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(top=0.80, wspace=0.36)
    _brand(fig, "Model evolution")
    fig.suptitle(
        "Every upgrade had to beat the frozen baseline",
        x=0.025,
        y=0.93,
        ha="left",
        fontsize=27,
        weight="bold",
    )
    fig.text(
        0.025,
        0.865,
        "Original enrichment + environment + events + announced lineups + CUDA architectures",
        color=MUTED,
        fontsize=11,
    )
    ax = axes[0]
    view = summary.sort_values("basis_points", ascending=False)
    colors = [
        CYAN if value < -10 else RED if value > 10 else BLUE
        for value in view["basis_points"]
    ]
    bars = ax.barh(view["configuration"], view["basis_points"], color=colors)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title("Probability impact", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Log-loss change · basis points")
    ax.grid(axis="x", alpha=0.45)
    for bar, value in zip(bars, view["basis_points"]):
        ax.text(
            value + (1 if value >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color=TEXT,
            fontsize=9,
        )

    ax = axes[1]
    ax.scatter(
        summary["log_loss_delta"] * 10_000,
        summary["accuracy_delta"] * 100,
        s=130,
        c=[
            CYAN if loss < 0 and accuracy >= 0 else GOLD if loss < 0 else RED
            for loss, accuracy in zip(
                summary["log_loss_delta"], summary["accuracy_delta"]
            )
        ],
        edgecolor=NAVY,
    )
    annotate = summary.loc[
        summary["configuration"].isin(
            [
                "Baseline scores + state",
                "+ shootout dynamics",
                "+ StatsBomb event dynamics",
                "+ oracle weather",
                "CUDA XGBoost",
                "CUDA residual network",
            ]
        )
    ]
    for row in annotate.itertuples():
        short_name = row.configuration.replace(
            "Baseline scores + state", "Baseline"
        )
        ax.annotate(
            short_name,
            (row.log_loss_delta * 10_000, row.accuracy_delta * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=TEXT,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("Promotion frontier", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Log-loss change · basis points ← better")
    ax.set_ylabel("Accuracy change · percentage points")
    ax.grid(alpha=0.45)
    fig.text(
        0.025,
        0.025,
        "Unified June 21 evidence: zero staged candidates passed the full production promotion gate.",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "model-evolution.png")


def generate_visual_report(
    backtests: pd.DataFrame,
    benchmarks: pd.DataFrame,
    reliability: pd.DataFrame,
    ablation: pd.DataFrame,
    simulation: pd.DataFrame,
    fixtures: pd.DataFrame,
    ratings: pd.DataFrame,
    turkey_schedule: pd.DataFrame | None = None,
    turkey_scenarios: pd.DataFrame | None = None,
    turkey_blind: pd.DataFrame | None = None,
    tournament_blind: pd.DataFrame | None = None,
    blind_leaderboard: pd.DataFrame | None = None,
    enrichment: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
) -> list[Path]:
    paths = [
        hero_background(),
        model_evidence_chart(backtests, benchmarks, reliability, ablation),
        tournament_forecast_chart(simulation),
        team_power_map(ratings, simulation["team"].tolist()),
        forecast_social_card(simulation, backtests, fixtures),
    ]
    if turkey_schedule is not None and turkey_scenarios is not None:
        paths.append(turkey_focus_chart(turkey_schedule, turkey_scenarios, simulation))
    if turkey_blind is not None:
        paths.append(turkey_blind_chart(turkey_blind))
    if tournament_blind is not None and blind_leaderboard is not None:
        paths.append(
            blind_tournament_chart(tournament_blind, blind_leaderboard)
        )
    if enrichment is not None and coverage is not None:
        paths.append(model_evolution_chart(enrichment, coverage))
    return paths


def generate_unified_visual_report(
    backtests: pd.DataFrame,
    benchmarks: pd.DataFrame,
    reliability: pd.DataFrame,
    ablation: pd.DataFrame,
    simulation: pd.DataFrame,
    fixtures: pd.DataFrame,
    ratings: pd.DataFrame,
    turkey_schedule: pd.DataFrame,
    turkey_scenarios: pd.DataFrame,
    turkey_blind: pd.DataFrame,
    tournament_blind: pd.DataFrame,
    blind_leaderboard: pd.DataFrame,
    enrichment: pd.DataFrame,
    coverage: pd.DataFrame,
    advanced: pd.DataFrame,
    architectures: pd.DataFrame,
    squad_teams: pd.DataFrame,
    squad_players: pd.DataFrame,
) -> list[Path]:
    """Regenerate every publication asset with the unified research status."""
    production_note = (
        "Unified June 21 evidence · production remains calibrated hybrid · "
        "staged candidates are displayed but not silently blended"
    )
    audit_note = (
        "Immutable blind baseline · June 21 research is not retrofitted · "
        "preserved for version-to-version comparison"
    )
    turkey_squad = squad_teams.loc[squad_teams["fifa_code"].eq("TUR")].iloc[0]
    turkey_players = squad_players.loc[squad_players["fifa_code"].eq("TUR")]
    paths = [
        hero_background(),
        model_evidence_chart(
            backtests,
            benchmarks,
            reliability,
            ablation,
            production_note,
        ),
        tournament_forecast_chart(simulation, production_note),
        team_power_map(
            ratings,
            simulation["team"].tolist(),
            production_note + " · official squads available as context",
        ),
        forecast_social_card(
            simulation,
            backtests,
            fixtures,
            "Calibrated hybrid retained · 0 advanced candidates promoted · 48 official squads analyzed",
        ),
        turkey_focus_chart(
            turkey_schedule,
            turkey_scenarios,
            simulation,
            turkey_squad,
            turkey_players,
            production_note,
        ),
        turkey_blind_chart(turkey_blind, audit_note),
        blind_tournament_chart(
            tournament_blind,
            blind_leaderboard,
            audit_note,
        ),
        model_evolution_chart(enrichment, coverage, advanced),
        advanced_research_chart(advanced, architectures),
        squad_intelligence_chart(squad_teams),
    ]
    return paths

