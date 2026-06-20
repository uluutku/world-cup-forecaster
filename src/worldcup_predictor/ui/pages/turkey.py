from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_turkey_focus(
    *,
    simulation: pd.DataFrame,
    schedule: pd.DataFrame,
    blind: pd.DataFrame,
    scenarios: pd.DataFrame,
    report: dict[str, object],
    squad_teams: pd.DataFrame,
    squad_players: pd.DataFrame,
    color: dict[str, str],
    metric_card: Callable[[str, str, str], None],
    section: Callable[[str, str, str], None],
    style_chart: Callable[[go.Figure, int], go.Figure],
) -> None:
    state = simulation.loc[simulation["team"].eq("Turkey")].iloc[0]
    remaining = schedule.loc[~schedule["is_completed"]].iloc[0]
    metadata = report["turkey_blind_experiment"]
    squad = squad_teams.loc[squad_teams["fifa_code"].eq("TUR")].iloc[0]
    players = squad_players.loc[
        squad_players["fifa_code"].eq("TUR")
    ].sort_values(["position", "number"])

    section(
        "Türkiye intelligence room",
        "Blind evidence, live forecast, and the official squad",
        "The blind audit is immutable. Current probabilities come from the promoted hybrid; squad and event research remain separate evidence layers.",
    )
    st.markdown(
        f"""
        <div class="evidence">
        <b>Blind time-machine experiment.</b> The model was frozen before Türkiye's first
        match on <b>June 12, 2026</b>. It used
        <b>{metadata['training_matches']:,}</b> earlier matches and consumed exactly
        <b>{metadata['state_updates_from_hidden_matches']}</b> hidden tournament results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    blind_long = blind.melt(
        id_vars=["date", "opponent"],
        value_vars=["p_team_win", "p_draw", "p_opponent_win"],
        var_name="outcome",
        value_name="probability",
    )
    blind_long["outcome"] = blind_long["outcome"].map(
        {
            "p_team_win": "Türkiye win",
            "p_draw": "Draw",
            "p_opponent_win": "Opponent win",
        }
    )
    fig = px.bar(
        blind_long,
        x="probability",
        y="opponent",
        color="outcome",
        orientation="h",
        barmode="stack",
        text=blind_long["probability"].map(lambda value: f"{value:.1%}"),
        color_discrete_map={
            "Türkiye win": color["cyan"],
            "Draw": color["gold"],
            "Opponent win": color["red"],
        },
        title="Frozen pre-tournament outcome probabilities",
        labels={"probability": "Outcome probability", "opponent": ""},
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)

    blind_display = blind[
        [
            "date",
            "opponent",
            "p_team_win",
            "p_draw",
            "p_opponent_win",
            "team_view_pick",
            "actual_result",
            "correct_pick",
        ]
    ].copy()
    st.dataframe(
        blind_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D"),
            "p_team_win": st.column_config.ProgressColumn(
                "Türkiye win", min_value=0, max_value=1, format="%.1%%"
            ),
            "p_draw": st.column_config.ProgressColumn(
                "Draw", min_value=0, max_value=1, format="%.1%%"
            ),
            "p_opponent_win": st.column_config.ProgressColumn(
                "Opponent win", min_value=0, max_value=1, format="%.1%%"
            ),
            "team_view_pick": "Blind pick",
            "actual_result": "Actual result",
            "correct_pick": "Correct?",
        },
    )

    section(
        "Official squad intelligence",
        "What the roster adds beyond match results",
        "FIFA caps, goals and heights are joined to age, position, club and league context for all 26 players.",
    )
    cards = st.columns(6)
    values = [
        ("Mean age", f"{squad['mean_age']:.1f}", f"{squad['under_23_share']:.0%} under 23"),
        ("Caps", f"{squad['total_caps']:,.0f}", "13th of 48 squads"),
        ("Intl goals", f"{squad['total_international_goals']:,.0f}", "current roster total"),
        ("Top-five leagues", f"{squad['top_five_league_share']:.0%}", "ENG · ESP · GER · ITA · FRA"),
        ("Domestic clubs", f"{squad['domestic_club_share']:.0%}", "share playing in Türkiye"),
        ("Data coverage", f"{squad['official_stats_coverage']:.0%}", "26 of 26 players joined"),
    ]
    for column, (label, value, note) in zip(cards, values):
        with column:
            metric_card(label, value, note)

    fingerprint_tab, players_tab, implications_tab = st.tabs(
        ["Squad fingerprint", "Player depth", "Forecast implications"]
    )
    with fingerprint_tab:
        metrics = {
            "Experience": "total_caps",
            "Scoring": "total_international_goals",
            "Top-five exposure": "top_five_league_share",
            "Club diversity": "club_diversity",
            "Forward depth": "forward_goal_depth",
            "Youth share": "under_23_share",
        }
        labels = list(metrics)
        percentiles = [
            float((squad_teams[column] <= float(squad[column])).mean())
            for column in metrics.values()
        ]
        left, right = st.columns([0.9, 1.1])
        with left:
            radar = go.Figure(
                go.Scatterpolar(
                    r=percentiles + [percentiles[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    line_color=color["cyan"],
                    fillcolor="rgba(69,230,194,.18)",
                )
            )
            radar.update_layout(
                title="Türkiye percentile versus 48 squads",
                showlegend=False,
                polar={
                    "radialaxis": {
                        "range": [0, 1],
                        "tickformat": ".0%",
                        "gridcolor": color["line"],
                    },
                    "angularaxis": {"gridcolor": color["line"]},
                    "bgcolor": "rgba(0,0,0,0)",
                },
            )
            st.plotly_chart(style_chart(radar, 520), use_container_width=True)
        with right:
            comparison = squad_teams.assign(
                focus=np.where(
                    squad_teams["fifa_code"].eq("TUR"),
                    "Türkiye",
                    "Other squad",
                )
            )
            scatter = px.scatter(
                comparison,
                x="mean_age",
                y="total_caps",
                size="total_international_goals",
                color="focus",
                hover_name="team",
                color_discrete_map={
                    "Türkiye": color["gold"],
                    "Other squad": color["blue"],
                },
                title="Age × experience × proven scoring",
                labels={
                    "mean_age": "Mean squad age",
                    "total_caps": "Combined international caps",
                },
            )
            st.plotly_chart(style_chart(scatter, 520), use_container_width=True)

    with players_tab:
        view = players.copy()
        view["experience_band"] = pd.cut(
            view["caps"].fillna(0),
            bins=[-1, 10, 30, 60, np.inf],
            labels=["Emerging", "Rotation", "Experienced", "Veteran"],
        )
        left, right = st.columns([1.05, 0.95])
        with left:
            scatter = px.scatter(
                view,
                x="age",
                y="caps",
                size="international_goals",
                color="position",
                hover_name="player",
                hover_data={
                    "club": True,
                    "height": ":.0f",
                    "international_goals": ":.0f",
                },
                color_discrete_map={
                    "GK": color["gold"],
                    "DF": color["blue"],
                    "MF": color["cyan"],
                    "FW": color["red"],
                },
                title="Age, experience and scoring contribution",
            )
            st.plotly_chart(style_chart(scatter, 560), use_container_width=True)
        with right:
            goals = view.nlargest(12, "international_goals").sort_values(
                "international_goals"
            )
            bars = px.bar(
                goals,
                x="international_goals",
                y="player",
                orientation="h",
                color="position",
                color_discrete_map={
                    "GK": color["gold"],
                    "DF": color["blue"],
                    "MF": color["cyan"],
                    "FW": color["red"],
                },
                title="International scoring depth",
                labels={"international_goals": "Goals", "player": ""},
            )
            st.plotly_chart(style_chart(bars, 560), use_container_width=True)
        st.dataframe(
            view[
                [
                    "number",
                    "position",
                    "player",
                    "age",
                    "height",
                    "caps",
                    "international_goals",
                    "club",
                    "club_country",
                    "experience_band",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "age": st.column_config.NumberColumn(format="%.1f"),
                "height": st.column_config.NumberColumn(format="%.0f cm"),
                "caps": st.column_config.NumberColumn(format="%.0f"),
                "international_goals": st.column_config.NumberColumn(
                    "Goals", format="%.0f"
                ),
            },
        )

    with implications_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Structural strengths")
            st.markdown("- 1,005 combined caps rank 13th among the 48 squads.")
            st.markdown(
                f"- {squad['under_23_share']:.0%} of the squad is under 23."
            )
            st.markdown("- All 26 official player records were joined without imputation.")
        with right:
            st.markdown("#### Structural risks")
            st.markdown(
                f"- {squad['top_five_league_share']:.0%} play in the five strongest European leagues."
            )
            st.markdown(
                f"- {squad['domestic_club_share']:.0%} of the squad plays domestically."
            )
            st.markdown("- International scoring is concentrated in a small group.")
        st.markdown(
            f"""
            <div class="evidence">
            <b>Interpretation:</b> these roster signals are not silently added to the
            {remaining['p_away']:.1%} Türkiye win probability. Historical squad features
            overfit, event data improved only one fold, and announced-lineup features
            regressed. They remain decision context until the promotion gate is passed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    blind_usa = blind.loc[
        blind["opponent"].eq("United States"), "p_team_win"
    ].iloc[0]
    live_change = remaining["p_away"] - blind_usa
    cards = st.columns(4)
    with cards[0]:
        metric_card(
            "Current R32 chance",
            f"{state['round_32']:.1%}",
            "live tournament simulation",
        )
    with cards[1]:
        metric_card(
            "Türkiye win",
            f"{remaining['p_away']:.1%}",
            "against the United States",
        )
    with cards[2]:
        metric_card(
            "Blind USA forecast",
            f"{blind_usa:.1%}",
            f"change after two losses {live_change:+.1%}",
        )
    with cards[3]:
        metric_card(
            "USA win",
            f"{remaining['p_home']:.1%}",
            "highest-probability outcome",
        )

    left, right = st.columns([1.05, 1])
    with left:
        section(
            "Schedule",
            "Every planned Türkiye match",
            "Completed fixtures show final results; the remaining fixture shows outcome probabilities.",
        )
        display = schedule[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "status",
                "p_home",
                "p_draw",
                "p_away",
            ]
        ].copy()
        display["home_team"] = display["home_team"].replace({"Turkey": "Türkiye"})
        display["away_team"] = display["away_team"].replace({"Turkey": "Türkiye"})
        final_result = np.select(
            [
                display["home_score"] > display["away_score"],
                display["home_score"] < display["away_score"],
                display["home_score"].eq(display["away_score"]),
            ],
            ["Home win", "Away win", "Draw"],
            default="",
        )
        forecast = (
            display["p_home"].map(lambda value: f"Home {value:.1%}")
            + " · Draw "
            + display["p_draw"].map(lambda value: f"{value:.1%}")
            + " · Away "
            + display["p_away"].map(lambda value: f"{value:.1%}")
        )
        display["result_or_forecast"] = np.where(
            display["status"].eq("Final"), final_result, forecast
        )
        st.dataframe(
            display[
                ["date", "home_team", "away_team", "status", "result_or_forecast"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="MMM D")
            },
        )
        bars = go.Figure(
            go.Bar(
                x=[
                    remaining["p_home"],
                    remaining["p_draw"],
                    remaining["p_away"],
                ],
                y=["United States win", "Draw", "Türkiye win"],
                orientation="h",
                marker_color=[color["red"], color["gold"], color["cyan"]],
                text=[
                    f"{remaining['p_home']:.1%}",
                    f"{remaining['p_draw']:.1%}",
                    f"{remaining['p_away']:.1%}",
                ],
                textposition="outside",
            )
        )
        bars.update_layout(title="United States vs Türkiye · June 25")
        bars.update_xaxes(tickformat=".0%", range=[0, 0.62])
        st.plotly_chart(style_chart(bars, 380), use_container_width=True)

    with right:
        section(
            "Survival scenarios",
            "Outcome-conditioned tournament paths",
            "The simulator integrates over the internal score process and reports only win, draw and loss conditions.",
        )
        scenario_view = scenarios.sort_values("round_32")
        bars = px.bar(
            scenario_view,
            x="round_32",
            y="scenario",
            orientation="h",
            color="round_32",
            text=scenario_view["round_32"].map(lambda value: f"{value:.1%}"),
            color_continuous_scale=[
                color["red"],
                color["gold"],
                color["cyan"],
            ],
            title="Probability of reaching the Round of 32",
            labels={"round_32": "Advance probability", "scenario": ""},
        )
        bars.update_layout(coloraxis_showscale=False)
        bars.update_xaxes(
            tickformat=".0%",
            range=[0, max(0.55, scenario_view["round_32"].max() * 1.15)],
        )
        st.plotly_chart(style_chart(bars, 430), use_container_width=True)
        win_advance = scenario_view.loc[
            scenario_view["scenario"].eq("Türkiye win"), "round_32"
        ].iloc[0]
        st.markdown(
            f"""
            <div class="evidence">
            <b>Model verdict:</b> Türkiye must win. Conditional on a victory, the
            estimated probability of reaching the Round of 32 is
            <b>{win_advance:.1%}</b>. A draw or loss produces 0% in the current group state.
            </div>
            """,
            unsafe_allow_html=True,
        )

    stages = scenarios.melt(
        id_vars=["scenario"],
        value_vars=["round_32", "quarterfinal", "semifinal", "final", "champion"],
        var_name="stage",
        value_name="probability",
    )
    stages = stages.loc[stages["scenario"].eq("Türkiye win")]
    curve = px.line(
        stages,
        x="stage",
        y="probability",
        markers=True,
        title="Türkiye tournament path conditional on a win",
        labels={"stage": "", "probability": "Reach probability"},
        color_discrete_sequence=[color["cyan"]],
    )
    curve.update_yaxes(tickformat=".1%")
    st.plotly_chart(style_chart(curve, 430), use_container_width=True)
