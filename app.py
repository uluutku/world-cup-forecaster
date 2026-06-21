from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from worldcup_predictor.config import VISUAL_DIR
from worldcup_predictor.features import FEATURE_COLUMNS
from worldcup_predictor.ui.components import (
    COLOR,
    FULL_WIDTH,
    image_data_uri,
    metric_card,
    probability_label,
    section,
    style_chart,
)
from worldcup_predictor.ui.data import load_runtime
from worldcup_predictor.ui.pages.turkey import render_turkey_focus

st.set_page_config(
    page_title="World Cup Intelligence · 2026 Forecasts",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

hero_image = image_data_uri(VISUAL_DIR / "hero-data-field.png")
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] {{ font-family: "Inter", sans-serif; }}
      .stApp {{
        background:
          radial-gradient(circle at 82% 3%, rgba(69,230,194,.09), transparent 27rem),
          radial-gradient(circle at 8% 34%, rgba(91,143,249,.08), transparent 30rem),
          {COLOR['ink']};
      }}
      [data-testid="stSidebar"] {{
        background: rgba(6,15,26,.96);
        border-right: 1px solid {COLOR['line']};
      }}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: {COLOR['muted']}; }}
      .block-container {{ max-width: 1540px; padding-top: 1.4rem; padding-bottom: 4rem; }}
      h1, h2, h3 {{ font-family: "Space Grotesk", sans-serif !important; letter-spacing: -.025em; }}
      .hero {{
        position: relative; overflow: hidden; min-height: 315px; border-radius: 24px;
        border: 1px solid {COLOR['line']}; padding: 3.2rem 3.4rem; margin-bottom: 1.2rem;
        background:
          linear-gradient(90deg, rgba(5,11,20,.98) 0%, rgba(5,11,20,.91) 45%, rgba(5,11,20,.20) 100%),
          url("{hero_image}") right center / cover no-repeat,
          linear-gradient(135deg, #091827, #10243a);
        box-shadow: 0 22px 80px rgba(0,0,0,.28);
      }}
      .hero::after {{
        content:""; position:absolute; inset:0; pointer-events:none;
        background: linear-gradient(120deg, transparent 35%, rgba(69,230,194,.05) 48%, transparent 60%);
      }}
      .kicker {{ color:{COLOR['cyan']}; font-size:.73rem; letter-spacing:.18em; font-weight:800; }}
      .hero h1 {{ color:{COLOR['text']}; font-size:3.65rem; line-height:.96; margin:.55rem 0 .8rem; max-width:670px; }}
      .hero p {{ color:{COLOR['muted']}; font-size:1.02rem; max-width:700px; line-height:1.65; }}
      .chip {{
        display:inline-block; border:1px solid #29435e; border-radius:99px; color:#bfd0df;
        padding:.4rem .72rem; margin:.8rem .35rem 0 0; font-size:.72rem; background:rgba(10,25,41,.72);
      }}
      .metric-card {{
        background:linear-gradient(145deg,rgba(14,28,45,.96),rgba(8,20,34,.95));
        border:1px solid {COLOR['line']}; border-radius:16px; padding:1.05rem 1.15rem;
        min-height:126px; position:relative; overflow:hidden;
      }}
      .metric-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:{COLOR['cyan']}; }}
      .metric-label {{ color:{COLOR['muted']}; font-size:.69rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }}
      .metric-value {{ color:{COLOR['text']}; font:700 1.78rem "Space Grotesk"; margin:.38rem 0 .22rem; }}
      .metric-note {{ color:{COLOR['cyan']}; font-size:.76rem; line-height:1.35; }}
      .section-kicker {{ color:{COLOR['cyan']}; text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:800; margin-top:1rem; }}
      .section-title {{ color:{COLOR['text']}; font:700 1.75rem "Space Grotesk"; margin:.2rem 0 .15rem; }}
      .section-copy {{ color:{COLOR['muted']}; font-size:.88rem; margin-bottom:.75rem; }}
      .evidence {{
        border:1px solid {COLOR['line']}; border-radius:14px; padding:1rem 1.1rem;
        background:rgba(13,28,45,.66); color:{COLOR['muted']}; line-height:1.55; font-size:.85rem;
      }}
      .evidence b {{ color:{COLOR['text']}; }}
      div[data-testid="stDataFrame"] {{ border:1px solid {COLOR['line']}; border-radius:14px; overflow:hidden; }}
      [data-testid="stMetric"] {{ background:{COLOR['panel']}; border:1px solid {COLOR['line']}; padding:1rem; border-radius:14px; }}
      .stButton > button, .stDownloadButton > button {{
        border-radius:10px; border:1px solid #2b4a67; background:#0d2136; color:{COLOR['text']};
      }}
      .stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{COLOR['cyan']}; color:{COLOR['cyan']}; }}
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ background:#0b1a2b; border-color:{COLOR['line']}; }}
      hr {{ border-color:{COLOR['line']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


bundle, data = load_runtime()
model, builder = bundle["model"], bundle["builder"]
report = data["report"]
fixtures = data["fixtures"]
simulation = data["simulation"]
backtest_predictions = data["backtest_predictions"]
benchmarks = data["benchmarks"]
ablation = data["ablation"]
drift = data["drift"]
enrichment = data["enrichment"]
coverage = data["coverage"]
advanced_report = data["advanced_report"]
advanced_matrix = data["advanced_matrix"]
architectures = data["architectures"]
lineup_benchmark = data["lineup_benchmark"]
squad_teams = data["squad_teams"]
squad_players = data["squad_players"]
tournament_blind = data["tournament_blind"]
blind_leaderboard = data["blind_leaderboard"]
turkey_schedule = data["turkey"]
turkey_blind = data["turkey_blind"]
turkey_scenarios = data["turkey_scenarios"]
backtests = pd.DataFrame(report["backtests"])
ratings = pd.DataFrame(report["ratings"])
embedding = pd.DataFrame(report["team_embedding"])
reliability = pd.DataFrame(report["reliability"])

with st.sidebar:
    st.markdown("### ◉ WCI")
    st.caption("PROBABILISTIC FOOTBALL RESEARCH")
    st.divider()
    page = st.radio(
        "Workspace",
        [
            "Command Center",
            "Türkiye Focus",
            "Blind Tournament Audit",
            "Model Evolution",
            "Advanced Laboratory",
            "Tournament Engine",
            "Match Laboratory",
            "Model Observatory",
            "Team Atlas",
            "Research Artifacts",
        ],
        index=1,
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Experiment lineage**")
    st.caption(report["experiment"]["id"])
    st.caption(f"Data hash · {report['experiment']['data_sha256'][:12]}")
    st.caption(f"Last result · {report['data']['last_result']}")
    st.caption(f"{report['data']['completed_matches']:,} observed matches")
    st.divider()
    st.markdown("**Ensemble allocation**")
    for name, weight in report["model"]["blend_weights"].items():
        st.progress(weight, text=f"{name.replace('_', ' ').title()} · {weight:.0%}")
    st.caption(f"Dixon–Coles ρ · {report['model']['dixon_coles_rho']:+.2f}")
    st.caption(f"Calibration T · {report['model']['temperature']:.2f}")

st.markdown(
    """
    <div class="hero">
      <div class="kicker">WORLD CUP 2026 FORECASTS</div>
      <h1>World Cup<br/>Intelligence</h1>
      <p>A football model that predicts match outcomes, plays out the whole 2026 tournament
      thousands of times, and shows you how sure it really is. It only ever uses what was
      known before kickoff.</p>
      <span class="chip">No future data</span><span class="chip">Dixon–Coles</span>
      <span class="chip">Conformal sets</span><span class="chip">Walk-forward validation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

favorite = simulation.iloc[0]
mean_loss = backtests["log_loss"].mean()
mean_baseline = backtests["baseline_log_loss"].mean()
skill = 1 - mean_loss / mean_baseline
bootstrap = report["bootstrap_intervals"]
conformal = report["conformal"]
advanced_generated = pd.Timestamp(advanced_report["generated_at"]).tz_convert(
    "Europe/Istanbul"
)
promoted_candidates = int(
    advanced_matrix["decision"].eq("Promotion candidate").sum()
)
event_research = advanced_matrix.loc[
    advanced_matrix["candidate"].eq("+ StatsBomb event dynamics")
].iloc[0]
turkey_squad = squad_teams.loc[squad_teams["fifa_code"].eq("TUR")].iloc[0]

st.markdown(
    f"""
    <div class="evidence">
    <b>Unified evidence state · {advanced_generated.strftime('%d %B %Y')}</b> &nbsp;|&nbsp;
    Production remains the calibrated hybrid because <b>{promoted_candidates} staged candidates</b>
    passed the promotion gate. CUDA, weather, travel, StatsBomb event, announced-lineup, and official
    squad intelligence are now visible throughout the product without being silently mixed into
    production probabilities. Latest maintained result: <b>{report['data']['last_result']}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Command Center":
    cols = st.columns(4)
    with cols[0]:
        metric_card("Title favorite", favorite["team"], f"{favorite['champion']:.1%} simulated champion probability")
    with cols[1]:
        metric_card("Probabilistic skill", f"{skill:.1%}", "log-loss improvement over historical prior")
    with cols[2]:
        metric_card("90% conformal coverage", f"{conformal['empirical_coverage']:.1%}", f"mean set size {conformal['average_set_size']:.2f}")
    with cols[3]:
        metric_card("Backtest uncertainty", f"{mean_loss:.3f}", f"95% CI {bootstrap['log_loss']['low']:.3f}–{bootstrap['log_loss']['high']:.3f}")

    research_cols = st.columns(4)
    with research_cols[0]:
        metric_card(
            "Advanced promotion gate",
            f"{promoted_candidates} passed",
            "production probabilities remain statistically controlled",
        )
    with research_cols[1]:
        metric_card(
            "Best staged signal",
            f"{event_research['log_loss_basis_points']:+.1f} bp",
            "StatsBomb events · 2022 only · accuracy tradeoff",
        )
    with research_cols[2]:
        metric_card(
            "Official squad layer",
            "48 / 48 teams",
            f"{advanced_report['coverage']['official_squad_join_rate']:.1%} player-stat join",
        )
    with research_cols[3]:
        metric_card(
            "GPU verdict",
            "Hybrid retained",
            f"{advanced_report['runtime']['cuda_device']} benchmarked",
        )

    section(
        "Live forecast",
        "Where things stand right now",
        "These chances mix the 2026 results played so far with 5,000 simulations of the games still to come.",
    )
    left, right = st.columns([1.25, 1])
    with left:
        top = simulation.head(18).sort_values("champion")
        fig = px.bar(
            top,
            x="champion",
            y="team",
            orientation="h",
            color="champion",
            color_continuous_scale=["#244363", COLOR["blue"], COLOR["cyan"]],
            labels={"champion": "Champion probability", "team": ""},
            title="Title probability",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, 570), **FULL_WIDTH)
    with right:
        stages = simulation.head(12).set_index("team")[
            ["round_32", "quarterfinal", "semifinal", "final", "champion"]
        ]
        fig = go.Figure(
            go.Heatmap(
                z=stages.values,
                x=["R32", "Quarterfinal", "Semifinal", "Final", "Champion"],
                y=stages.index,
                colorscale=[[0, "#0c1a2b"], [0.45, "#2b5e83"], [1, COLOR["cyan"]]],
                text=np.vectorize(lambda value: f"{value:.0%}")(stages.values),
                texttemplate="%{text}",
                hovertemplate="%{y} · %{x}<br>%{z:.1%}<extra></extra>",
            )
        )
        fig.update_layout(title="Tournament survival matrix")
        st.plotly_chart(style_chart(fig, 570), **FULL_WIDTH)

    section(
        "Decision surface",
        "Where the next fixtures are hardest to call",
        "Entropy is how uncertain the outcome is; disagreement is how much the three models argue with each other.",
    )
    uncertain = fixtures.nlargest(12, "entropy").copy()
    fig = px.scatter(
        uncertain,
        x="entropy",
        y="model_disagreement",
        size="total_xg",
        color="confidence",
        hover_name="home_team",
        hover_data={"away_team": True, "p_home": ":.1%", "p_draw": ":.1%", "p_away": ":.1%"},
        text=uncertain["home_team"] + " · " + uncertain["away_team"],
        color_continuous_scale=["#ef6a72", "#f5c85b", "#45e6c2"],
        title="Uncertainty × model disagreement",
        labels={"entropy": "Predictive entropy", "model_disagreement": "Model-family disagreement", "confidence": "Confidence"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(style_chart(fig, 490), **FULL_WIDTH)

elif page == "Türkiye Focus":
    render_turkey_focus(
        simulation=simulation,
        schedule=turkey_schedule,
        blind=turkey_blind,
        scenarios=turkey_scenarios,
        report=report,
        squad_teams=squad_teams,
        squad_players=squad_players,
        color=COLOR,
        metric_card=metric_card,
        section=section,
        style_chart=style_chart,
    )
elif page == "Blind Tournament Audit":
    blind_meta = report["tournament_blind_experiment"]
    completed_blind = tournament_blind.loc[tournament_blind["is_completed"]].copy()
    section(
        "Frozen before the opener",
        "A blind audit for every 2026 team",
        "The model was frozen on June 10, 2026 and predicted all 72 group fixtures. No World Cup result updates ratings, form, calibration, or features.",
    )
    st.markdown(
        """
        <div class="evidence">
        <b>Immutable by design:</b> the June 21 weather, event, lineup, CUDA, and official-squad
        research is not retrofitted into this ledger. This tab remains the clean baseline against
        which future model versions can be compared.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    with cards[0]:
        metric_card(
            "Scored matches",
            f"{blind_meta['scored_matches']}",
            f"{blind_meta['scheduled_matches']} predictions permanently timestamped",
        )
    with cards[1]:
        metric_card(
            "Blind accuracy",
            f"{blind_meta['accuracy']:.1%}",
            "three-way outcome classification",
        )
    with cards[2]:
        metric_card(
            "Blind log loss",
            f"{blind_meta['log_loss']:.3f}",
            "probability quality on completed matches",
        )
    with cards[3]:
        metric_card(
            "Tournament updates",
            str(blind_meta["state_updates_from_tournament"]),
            "strictly zero result leakage",
        )

    left, right = st.columns([1.25, 1])
    with left:
        blind_plot = completed_blind.copy()
        blind_plot["fixture"] = (
            blind_plot["home_team"] + " · " + blind_plot["away_team"]
        )
        blind_plot["confidence"] = blind_plot[["p_home", "p_draw", "p_away"]].max(
            axis=1
        )
        fig = px.scatter(
            blind_plot,
            x="confidence",
            y="actual_outcome_probability",
            color="correct_pick",
            size="confidence",
            hover_name="fixture",
            hover_data={
                "model_pick": True,
                "actual_outcome": True,
            },
            color_discrete_map={True: COLOR["cyan"], False: COLOR["red"]},
            title="Confidence versus probability assigned to reality",
            labels={
                "confidence": "Model confidence",
                "actual_outcome_probability": "Probability assigned to actual outcome",
                "correct_pick": "Correct pick",
            },
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, 500), **FULL_WIDTH)
    with right:
        section(
            "Team lens",
            "Blind scorecard by national team",
            "Most teams have only played one or two games so far, so treat this as a quick check rather than a ranking.",
        )
        leaderboard_view = blind_leaderboard.copy()
        leaderboard_view["team"] = leaderboard_view["team"].replace(
            {"Turkey": "Türkiye"}
        )
        leaderboard_view = leaderboard_view.sort_values(
            ["accuracy", "log_loss"], ascending=[False, True]
        )
        st.dataframe(
            leaderboard_view,
            **FULL_WIDTH,
            hide_index=True,
            height=470,
            column_config={
                "accuracy": st.column_config.ProgressColumn(
                    "Accuracy", min_value=0, max_value=1, format="%.0%%"
                ),
                "log_loss": st.column_config.NumberColumn(
                    "Log loss", format="%.3f"
                ),
                "brier": st.column_config.NumberColumn("Brier", format="%.3f"),
                "mean_actual_probability": st.column_config.ProgressColumn(
                    "Reality probability", min_value=0, max_value=1, format="%.1%%"
                ),
            },
        )

    section(
        "Prediction ledger",
        "Every timestamped group forecast",
        "Completed matches show the realized result and whether the highest-probability outcome was correct. Future matches remain untouched.",
    )
    ledger = tournament_blind.copy()
    ledger["fixture"] = ledger["home_team"] + " vs " + ledger["away_team"]
    ledger["actual_score"] = np.where(
        ledger["is_completed"],
        ledger["home_score"].fillna(0).astype(int).astype(str)
        + "–"
        + ledger["away_score"].fillna(0).astype(int).astype(str),
        "Pending",
    )
    st.dataframe(
        ledger[
            [
                "date",
                "fixture",
                "p_home",
                "p_draw",
                "p_away",
                "model_pick",
                "actual_score",
                "actual_outcome",
                "correct_pick",
            ]
        ],
        **FULL_WIDTH,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D"),
            "p_home": st.column_config.ProgressColumn(
                "Home win", min_value=0, max_value=1, format="%.1%%"
            ),
            "p_draw": st.column_config.ProgressColumn(
                "Draw", min_value=0, max_value=1, format="%.1%%"
            ),
            "p_away": st.column_config.ProgressColumn(
                "Away win", min_value=0, max_value=1, format="%.1%%"
            ),
        },
    )

elif page == "Model Evolution":
    section(
        "Controlled feature research",
        "What each new data family actually changed",
        "Every row uses identical World Cup cutoffs, model class, recency weighting, and metrics. Negative log-loss delta is an improvement.",
    )
    summary = (
        enrichment.groupby("configuration", as_index=False)
        .agg(
            log_loss=("log_loss", "mean"),
            accuracy=("accuracy", "mean"),
            brier=("brier", "mean"),
            rps=("rps", "mean"),
            ece=("ece", "mean"),
            log_loss_delta=("log_loss_delta", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
            features=("features", "max"),
        )
        .sort_values("log_loss")
    )
    summary["log_loss_basis_points"] = summary["log_loss_delta"] * 10_000
    summary["accuracy_points"] = summary["accuracy_delta"] * 100
    summary["decision"] = np.select(
        [
            summary["log_loss_delta"].lt(-0.001)
            & summary["accuracy_delta"].ge(0),
            summary["log_loss_delta"].lt(-0.001),
            summary["log_loss_delta"].gt(0.001),
        ],
        [
            "Promotion candidate",
            "Probability gain / accuracy tradeoff",
            "Rejected",
        ],
        default="Inconclusive",
    )
    best = summary.iloc[0]
    baseline_row = summary.loc[
        summary["configuration"].eq("Baseline scores + state")
    ].iloc[0]
    cards = st.columns(4)
    with cards[0]:
        metric_card(
            "Frozen production",
            "Baseline v2",
            f"log loss {baseline_row['log_loss']:.3f} · accuracy {baseline_row['accuracy']:.1%}",
        )
    with cards[1]:
        metric_card(
            "Best probability model",
            best["configuration"].replace("+ ", ""),
            f"{best['log_loss_basis_points']:+.1f} log-loss basis points",
        )
    with cards[2]:
        metric_card(
            "Accuracy change",
            f"{best['accuracy_points']:+.2f} pp",
            "best probability model versus baseline",
        )
    with cards[3]:
        promotion_count = int(summary["decision"].eq("Promotion candidate").sum())
        metric_card(
            "Promotions",
            str(promotion_count),
            "baseline remains production until both criteria improve",
        )

    left, right = st.columns(2)
    with left:
        ordered = summary.sort_values("log_loss_basis_points", ascending=False)
        fig = px.bar(
            ordered,
            x="log_loss_basis_points",
            y="configuration",
            orientation="h",
            color="decision",
            text=ordered["log_loss_basis_points"].map(lambda value: f"{value:+.1f}"),
            color_discrete_map={
                "Promotion candidate": COLOR["cyan"],
                "Probability gain / accuracy tradeoff": COLOR["gold"],
                "Rejected": COLOR["red"],
                "Inconclusive": COLOR["blue"],
            },
            title="Log-loss change · basis points",
            labels={
                "log_loss_basis_points": "Change versus baseline (lower is better)",
                "configuration": "",
            },
        )
        fig.add_vline(x=0, line_color=COLOR["muted"])
        st.plotly_chart(style_chart(fig, 600), **FULL_WIDTH)
    with right:
        fig = px.scatter(
            summary,
            x="log_loss",
            y="accuracy",
            color="decision",
            size="features",
            text="configuration",
            color_discrete_map={
                "Promotion candidate": COLOR["cyan"],
                "Probability gain / accuracy tradeoff": COLOR["gold"],
                "Rejected": COLOR["red"],
                "Inconclusive": COLOR["blue"],
            },
            title="Probability quality versus outcome accuracy",
            labels={"log_loss": "Mean log loss", "accuracy": "Accuracy"},
        )
        fig.update_traces(textposition="top center")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, 600), **FULL_WIDTH)

    section(
        "Stage-three matrix",
        "Environment, events, lineups and CUDA architectures",
        "The original enrichment benchmark and the latest advanced experiments now share one promotion vocabulary.",
    )
    advanced_view = advanced_matrix.loc[
        ~advanced_matrix["candidate"].isin(
            ["Baseline scores + state", "Calibrated hybrid"]
        )
    ].copy()
    advanced_view["candidate_label"] = (
        advanced_view["stage"]
        + " · "
        + advanced_view["candidate"].str.replace("+ ", "", regex=False)
    )
    left, right = st.columns([1.2, 0.8])
    with left:
        ordered = advanced_view.sort_values(
            "log_loss_basis_points", ascending=False
        )
        fig = px.bar(
            ordered,
            x="log_loss_basis_points",
            y="candidate_label",
            orientation="h",
            color="decision",
            text=ordered["log_loss_basis_points"].map(
                lambda value: f"{value:+.1f}"
            ),
            color_discrete_map={
                "Promotion candidate": COLOR["cyan"],
                "Coverage-limited": COLOR["gold"],
                "Oracle only": COLOR["blue"],
                "Rejected": COLOR["red"],
            },
            title="Advanced-stage log-loss impact",
            labels={
                "log_loss_basis_points": "Change versus relevant baseline · bp",
                "candidate_label": "",
            },
        )
        fig.add_vline(x=0, line_color=COLOR["muted"])
        st.plotly_chart(style_chart(fig, 590), **FULL_WIDTH)
    with right:
        status_counts = (
            advanced_view.groupby("decision", as_index=False)
            .size()
            .rename(columns={"size": "experiments"})
        )
        fig = px.pie(
            status_counts,
            names="decision",
            values="experiments",
            hole=0.62,
            color="decision",
            color_discrete_map={
                "Promotion candidate": COLOR["cyan"],
                "Coverage-limited": COLOR["gold"],
                "Oracle only": COLOR["blue"],
                "Rejected": COLOR["red"],
            },
            title="Research disposition",
        )
        st.plotly_chart(style_chart(fig, 590), **FULL_WIDTH)

    section(
        "Temporal stability",
        "Did the feature help every World Cup?",
        "The heatmap prevents one unusually favorable tournament from masquerading as a general improvement.",
    )
    heatmap = enrichment.pivot(
        index="configuration", columns="edition", values="log_loss_delta"
    )
    fig = go.Figure(
        go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.astype(str),
            y=heatmap.index,
            colorscale=[
                [0, COLOR["cyan"]],
                [0.5, "#15283d"],
                [1, COLOR["red"]],
            ],
            zmid=0,
            text=np.vectorize(lambda value: f"{value:+.3f}")(heatmap.values),
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}<br>Δ log loss %{z:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(title="Per-edition log-loss change versus baseline")
    st.plotly_chart(style_chart(fig, 590), **FULL_WIDTH)

    section(
        "Coverage registry",
        "What data entered the laboratory",
        "Coverage and timestamp boundaries are product features: sparse or stale sources are never treated as universally available.",
    )
    st.dataframe(
        coverage,
        **FULL_WIDTH,
        hide_index=True,
        column_config={
            "coverage": st.column_config.ProgressColumn(
                "Coverage", min_value=0, max_value=1, format="%.1%%"
            )
        },
    )
    st.markdown(
        """
        <div class="evidence">
        <b>Unified promotion decision:</b> no original or advanced candidate currently improves both
        primary log loss and outcome accuracy across sufficient chronological folds. Shootout history
        and StatsBomb events improve probability quality under limited conditions, but both lose
        classification accuracy. Weather, travel, announced lineups, CUDA XGBoost, the residual
        network, and historical squad composition are rejected. Baseline v2 remains production.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Advanced Laboratory":
    section(
        "Stage-three research",
        "Weather, event data, lineups, squads and CUDA models",
        "Every idea is tested against the same frozen baseline, and sorted by when its data is actually known before kickoff.",
    )
    hybrid = advanced_matrix.loc[
        (advanced_matrix["stage"].eq("Architecture"))
        & advanced_matrix["candidate"].eq("Calibrated hybrid")
    ].iloc[0]
    event_candidate = advanced_matrix.loc[
        advanced_matrix["candidate"].eq("+ StatsBomb event dynamics")
    ].iloc[0]
    weather_candidate = advanced_matrix.loc[
        (advanced_matrix["stage"].eq("Environment"))
        & advanced_matrix["candidate"].eq("+ oracle weather")
    ].iloc[0]
    cards = st.columns(4)
    with cards[0]:
        metric_card(
            "Production benchmark",
            f"{hybrid['log_loss']:.3f}",
            f"{hybrid['accuracy']:.1%} accuracy · five World Cups",
        )
    with cards[1]:
        metric_card(
            "Best new signal",
            f"{event_candidate['log_loss_basis_points']:+.1f} bp",
            "StatsBomb event history · 2022 only · not promoted",
        )
    with cards[2]:
        metric_card(
            "Weather result",
            f"{weather_candidate['log_loss_basis_points']:+.1f} bp",
            "ERA5 oracle weather regressed versus baseline",
        )
    with cards[3]:
        metric_card(
            "Official squad join",
            f"{advanced_report['coverage']['official_squad_join_rate']:.1%}",
            f"{advanced_report['coverage']['official_2026_squad_players']:,} players · 48 teams",
        )

    experiment_tab, architecture_tab, squad_tab, protocol_tab = st.tabs(
        [
            "Experiment matrix",
            "CUDA architectures",
            "2026 squad intelligence",
            "Availability protocol",
        ]
    )
    with experiment_tab:
        view = advanced_matrix.loc[
            ~advanced_matrix["candidate"].isin(
                ["Baseline scores + state", "Calibrated hybrid"]
            )
        ].copy()
        view["candidate_label"] = (
            view["stage"] + " · " + view["candidate"].str.replace("+ ", "", regex=False)
        )
        decision_colors = {
            "Promotion candidate": COLOR["cyan"],
            "Coverage-limited": COLOR["gold"],
            "Oracle only": COLOR["blue"],
            "Rejected": COLOR["red"],
        }
        left, right = st.columns([1.15, 0.85])
        with left:
            ordered = view.sort_values("log_loss_basis_points", ascending=False)
            fig = px.bar(
                ordered,
                x="log_loss_basis_points",
                y="candidate_label",
                orientation="h",
                color="decision",
                text=ordered["log_loss_basis_points"].map(
                    lambda value: f"{value:+.1f}"
                ),
                color_discrete_map=decision_colors,
                title="Log-loss change versus the relevant frozen baseline",
                labels={
                    "log_loss_basis_points": "Basis points · lower is better",
                    "candidate_label": "",
                },
            )
            fig.add_vline(x=0, line_color=COLOR["muted"])
            st.plotly_chart(style_chart(fig, 650), **FULL_WIDTH)
        with right:
            fig = px.scatter(
                view,
                x="log_loss_basis_points",
                y="accuracy_points",
                color="decision",
                symbol="stage",
                size="folds",
                hover_name="candidate",
                color_discrete_map=decision_colors,
                title="Promotion frontier",
                labels={
                    "log_loss_basis_points": "Δ log loss · bp",
                    "accuracy_points": "Δ accuracy · pp",
                },
            )
            fig.add_vline(x=0, line_color=COLOR["muted"])
            fig.add_hline(y=0, line_color=COLOR["muted"])
            st.plotly_chart(style_chart(fig, 650), **FULL_WIDTH)
        display_matrix = advanced_matrix[
            [
                "stage",
                "candidate",
                "evaluation_scope",
                "log_loss",
                "accuracy",
                "log_loss_basis_points",
                "accuracy_points",
                "availability",
                "decision",
            ]
        ].copy()
        st.dataframe(
            display_matrix,
            **FULL_WIDTH,
            hide_index=True,
            column_config={
                "log_loss": st.column_config.NumberColumn(
                    "Log loss", format="%.4f"
                ),
                "accuracy": st.column_config.NumberColumn(
                    "Accuracy", format="%.1%%"
                ),
                "log_loss_basis_points": st.column_config.NumberColumn(
                    "Δ loss · bp", format="%+.1f"
                ),
                "accuracy_points": st.column_config.NumberColumn(
                    "Δ accuracy · pp", format="%+.2f"
                ),
            },
        )

    with architecture_tab:
        architecture_summary = (
            architectures.groupby("architecture", as_index=False)
            .agg(
                log_loss=("log_loss", "mean"),
                accuracy=("accuracy", "mean"),
                training_seconds=("training_seconds", "mean"),
                parameters=("parameters", "max"),
            )
            .sort_values("log_loss")
        )
        left, right = st.columns([1.1, 0.9])
        with left:
            fig = px.line(
                architectures,
                x="edition",
                y="log_loss",
                color="architecture",
                markers=True,
                color_discrete_map={
                    "Calibrated hybrid": COLOR["cyan"],
                    "CUDA XGBoost": COLOR["blue"],
                    "CUDA residual network": COLOR["gold"],
                },
                title="Architecture stability by World Cup",
            )
            st.plotly_chart(style_chart(fig, 520), **FULL_WIDTH)
        with right:
            fig = px.bar(
                architecture_summary.sort_values("log_loss", ascending=False),
                x="log_loss",
                y="architecture",
                orientation="h",
                color="architecture",
                text=architecture_summary.sort_values(
                    "log_loss", ascending=False
                )["log_loss"].map(lambda value: f"{value:.3f}"),
                color_discrete_map={
                    "Calibrated hybrid": COLOR["cyan"],
                    "CUDA XGBoost": COLOR["blue"],
                    "CUDA residual network": COLOR["gold"],
                },
                title="Five-fold mean log loss",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_chart(fig, 520), **FULL_WIDTH)
        st.dataframe(
            architecture_summary,
            **FULL_WIDTH,
            hide_index=True,
            column_config={
                "log_loss": st.column_config.NumberColumn(format="%.4f"),
                "accuracy": st.column_config.NumberColumn(format="%.1%%"),
                "training_seconds": st.column_config.NumberColumn(
                    "Mean train seconds", format="%.2f"
                ),
                "parameters": st.column_config.NumberColumn(format="%,.0f"),
            },
        )
        st.markdown(
            f"""
            <div class="evidence">
            <b>GPU verification:</b> PyTorch {advanced_report['runtime']['torch']} sees
            <b>{advanced_report['runtime']['cuda_device']}</b>. CUDA XGBoost and the
            residual tabular network trained successfully. Both are rejected because
            their five-fold probability forecasts are worse, not because GPU support failed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with squad_tab:
        teams = squad_teams["team"].sort_values().tolist()
        selected_team = st.selectbox(
            "Inspect official squad",
            teams,
            index=teams.index("Turkey"),
        )
        team = squad_teams.loc[squad_teams["team"].eq(selected_team)].iloc[0]
        team_players = squad_players.loc[
            squad_players["team"].eq(selected_team)
        ].sort_values(["position", "number"])
        squad_cards = st.columns(5)
        squad_values = [
            ("Mean age", f"{team['mean_age']:.1f}", f"{team['under_23_share']:.0%} under 23"),
            ("Experience", f"{team['total_caps']:,.0f}", "combined international caps"),
            ("Scoring", f"{team['total_international_goals']:,.0f}", "combined international goals"),
            ("Top-five leagues", f"{team['top_five_league_share']:.0%}", "ENG · ESP · GER · ITA · FRA"),
            ("Club diversity", f"{team['club_diversity']:.0%}", "unique clubs per squad place"),
        ]
        for column, (label, value, note) in zip(squad_cards, squad_values):
            with column:
                metric_card(label, value, note)
        percentile_metrics = {
            "Caps": "total_caps",
            "Goals": "total_international_goals",
            "Top-five exposure": "top_five_league_share",
            "Club diversity": "club_diversity",
            "Forward depth": "forward_goal_depth",
            "Youth share": "under_23_share",
        }
        labels = list(percentile_metrics)
        percentiles = [
            float(
                (
                    squad_teams[column]
                    <= float(team[column])
                ).mean()
            )
            for column in percentile_metrics.values()
        ]
        radar_labels = labels + [labels[0]]
        radar_values = percentiles + [percentiles[0]]
        left, right = st.columns([0.85, 1.15])
        with left:
            fig = go.Figure(
                go.Scatterpolar(
                    r=radar_values,
                    theta=radar_labels,
                    fill="toself",
                    line_color=COLOR["cyan"],
                    fillcolor="rgba(69,230,194,.18)",
                    name=selected_team,
                )
            )
            fig.update_layout(
                polar={
                    "radialaxis": {
                        "range": [0, 1],
                        "tickformat": ".0%",
                        "gridcolor": COLOR["line"],
                    },
                    "angularaxis": {"gridcolor": COLOR["line"]},
                    "bgcolor": "rgba(0,0,0,0)",
                },
                title=f"{selected_team} · percentile versus 48 squads",
                showlegend=False,
            )
            st.plotly_chart(style_chart(fig, 530), **FULL_WIDTH)
        with right:
            fig = px.scatter(
                squad_teams,
                x="mean_age",
                y="total_caps",
                size="total_international_goals",
                color="top_five_league_share",
                hover_name="team",
                color_continuous_scale=["#17324b", COLOR["blue"], COLOR["cyan"]],
                title="Squad age, experience and scoring depth",
                labels={
                    "mean_age": "Mean age",
                    "total_caps": "Total caps",
                    "top_five_league_share": "Top-five share",
                },
            )
            selected_row = squad_teams.loc[
                squad_teams["team"].eq(selected_team)
            ]
            fig.add_trace(
                go.Scatter(
                    x=selected_row["mean_age"],
                    y=selected_row["total_caps"],
                    mode="markers+text",
                    text=[selected_team],
                    textposition="top center",
                    marker={
                        "size": 20,
                        "color": COLOR["gold"],
                        "line": {"color": COLOR["text"], "width": 2},
                    },
                    name=selected_team,
                )
            )
            st.plotly_chart(style_chart(fig, 530), **FULL_WIDTH)
        st.dataframe(
            team_players[
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
                ]
            ],
            **FULL_WIDTH,
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

    with protocol_tab:
        st.markdown(
            f"""
            <div class="evidence">
            <b>Weather:</b> {advanced_report['methodology']['weather']}<br><br>
            <b>Starting lineups:</b> {advanced_report['methodology']['lineups']}<br><br>
            <b>Promotion gate:</b> {advanced_report['methodology']['promotion_gate']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        protocol = pd.DataFrame(
            [
                ["Travel / altitude", "Before kickoff", "5 World Cups", "Rejected"],
                ["ERA5 observed weather", "After the event", "5 World Cups", "Oracle only"],
                ["StatsBomb event profile", "Before kickoff", "2022 only", "Coverage-limited"],
                ["Official starting XI", "About 60 min before kickoff", "2022 only", "Coverage-limited"],
                ["Official 2026 squad", "Tournament snapshot", "Current only", "Intelligence layer"],
                ["Injuries / availability", "Provider timestamp required", "Not open historically", "Adapter boundary"],
                ["Club minutes / load", "Before call-up and kickoff", "Historical provider required", "Adapter boundary"],
            ],
            columns=["Feature family", "Availability", "Evaluation", "Status"],
        )
        st.dataframe(protocol, **FULL_WIDTH, hide_index=True)

elif page == "Tournament Engine":
    section(
        "Monte Carlo engine",
        "Every contender, every stage",
        "Inspect path dependency rather than reading title probability as a single-number power ranking.",
    )
    st.caption(
        "Simulation probabilities use the promoted hybrid only. Official squads are available as context; rejected or coverage-limited research candidates do not alter tournament paths."
    )
    selected = st.multiselect(
        "Compare contenders",
        simulation["team"].tolist(),
        default=simulation.head(7)["team"].tolist(),
        max_selections=12,
    )
    selected_frame = simulation.loc[simulation["team"].isin(selected)]
    melted = selected_frame.melt(
        id_vars=["team", "group"],
        value_vars=["round_32", "quarterfinal", "semifinal", "final", "champion"],
        var_name="stage",
        value_name="probability",
    )
    stage_order = ["round_32", "quarterfinal", "semifinal", "final", "champion"]
    melted["stage"] = pd.Categorical(melted["stage"], stage_order, ordered=True)
    fig = px.line(
        melted.sort_values("stage"),
        x="stage",
        y="probability",
        color="team",
        markers=True,
        title="Path survival curves",
        labels={"stage": "", "probability": "Reach probability"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_chart(fig, 520), **FULL_WIDTH)

    left, right = st.columns([1.2, 1])
    with left:
        section("Groups", "Qualification pressure", "Current group position is reflected through observed results.")
        group_view = simulation.sort_values(["group", "round_32"], ascending=[True, False])
        fig = px.bar(
            group_view,
            x="round_32",
            y="team",
            color="group",
            facet_col="group",
            facet_col_wrap=4,
            orientation="h",
            labels={"round_32": "R32 probability", "team": ""},
            title="Advance probability by group",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("group=", "Group ")))
        st.plotly_chart(style_chart(fig, 760), **FULL_WIDTH)
    with right:
        section("Path leverage", "Bracket efficiency", "Title conversion conditional on reaching the Round of 32.")
        leverage = simulation.copy()
        leverage["conversion"] = leverage["champion"] / leverage["round_32"].clip(lower=1e-6)
        leverage = leverage.nlargest(20, "conversion").sort_values("conversion")
        fig = px.bar(
            leverage,
            x="conversion",
            y="team",
            orientation="h",
            color="conversion",
            color_continuous_scale=["#294866", COLOR["gold"], COLOR["cyan"]],
            title="Championship conversion after qualification",
            labels={"conversion": "Conditional title probability", "team": ""},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, 760), **FULL_WIDTH)

elif page == "Match Laboratory":
    section(
        "Counterfactual lab",
        "Stress-test a matchup",
        "Apply transparent strength shocks to represent lineup news, injuries, tactical changes, or analyst judgment.",
    )
    teams = sorted(simulation["team"].tolist())
    controls = st.columns([1, 1, 0.75, 0.8, 0.8])
    with controls[0]:
        home = st.selectbox("Listed first", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with controls[1]:
        away_default = "Argentina" if "Argentina" in teams else teams[1]
        away = st.selectbox("Listed second", teams, index=teams.index(away_default))
    with controls[2]:
        neutral = st.toggle("Neutral venue", True)
    with controls[3]:
        home_shock = st.slider("First-team shock", -150, 150, 0, 10, help="Rating points")
    with controls[4]:
        away_shock = st.slider("Second-team shock", -150, 150, 0, 10, help="Rating points")

    if home == away:
        st.warning("Choose two different teams.")
    else:
        home_squad = squad_teams.loc[squad_teams["team"].eq(home)]
        away_squad = squad_teams.loc[squad_teams["team"].eq(away)]
        base_row = pd.DataFrame(
            [builder.make_features(home, away, pd.Timestamp("2026-07-01"), "FIFA World Cup", "United States", neutral)]
        )
        scenario_row = base_row.copy()
        scenario_row["elo_diff"] += (home_shock - away_shock) / 400.0
        scenario_row["prestige_diff"] += (home_shock - away_shock) / 800.0
        base_probabilities = model.predict_proba(base_row[FEATURE_COLUMNS])[0]
        probabilities = model.predict_proba(scenario_row[FEATURE_COLUMNS])[0]
        components = model.component_probabilities(scenario_row[FEATURE_COLUMNS])
        disagreement = np.std(np.stack(list(components.values())), axis=0).mean()

        cards = st.columns(4)
        with cards[0]:
            metric_card(f"{home} win", f"{probabilities[2]:.1%}", "calibrated outcome probability")
        with cards[1]:
            metric_card("Draw", f"{probabilities[1]:.1%}", f"score dependence ρ {model.dc_rho:+.2f}")
        with cards[2]:
            metric_card(f"{away} win", f"{probabilities[0]:.1%}", "calibrated outcome probability")
        with cards[3]:
            delta = probabilities.max() - base_probabilities.max()
            metric_card("Scenario verdict", probability_label(probabilities, home, away), f"confidence shift {delta:+.1%} · disagreement {disagreement:.3f}")

        if not home_squad.empty and not away_squad.empty:
            section(
                "Current roster context",
                "Official squad comparison",
                "This is descriptive intelligence, not an unvalidated probability adjustment. Use the explicit rating shocks above when testing a lineup or injury hypothesis.",
            )
            compare_columns = st.columns(5)
            metrics_to_compare = [
                ("Mean age", "mean_age", ".1f"),
                ("Total caps", "total_caps", ",.0f"),
                ("Intl goals", "total_international_goals", ",.0f"),
                ("Top-five share", "top_five_league_share", ".0%"),
                ("Domestic share", "domestic_club_share", ".0%"),
            ]
            for column, (label, metric, formatting) in zip(
                compare_columns, metrics_to_compare
            ):
                with column:
                    first = format(float(home_squad.iloc[0][metric]), formatting)
                    second = format(float(away_squad.iloc[0][metric]), formatting)
                    metric_card(label, f"{first} · {second}", f"{home} · {away}")

        left, right = st.columns([1.15, 1])
        with left:
            matrix = model.score_matrix(scenario_row[FEATURE_COLUMNS], max_goals=7)[0]
            fig = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=[str(value) for value in range(8)],
                    y=[str(value) for value in range(8)],
                    colorscale=[[0, "#091626"], [0.5, "#286486"], [1, COLOR["cyan"]]],
                    text=np.vectorize(lambda value: f"{value:.1%}" if value > 0.015 else "")(matrix),
                    texttemplate="%{text}",
                    hovertemplate=f"{home} %{{y}} – %{{x}} {away}<br>%{{z:.2%}}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Dixon–Coles score-process diagnostic",
                xaxis_title=f"{away} goals",
                yaxis_title=f"{home} goals",
            )
            st.plotly_chart(style_chart(fig, 520), **FULL_WIDTH)
        with right:
            component_rows = []
            for name, values in components.items():
                component_rows.extend(
                    [
                        {"model": name, "outcome": f"{away} win", "probability": values[0, 0]},
                        {"model": name, "outcome": "Draw", "probability": values[0, 1]},
                        {"model": name, "outcome": f"{home} win", "probability": values[0, 2]},
                    ]
                )
            fig = px.bar(
                pd.DataFrame(component_rows),
                x="probability",
                y="model",
                color="outcome",
                orientation="h",
                barmode="group",
                color_discrete_sequence=[COLOR["red"], COLOR["gold"], COLOR["cyan"]],
                title="Model-family vote",
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_chart(fig, 520), **FULL_WIDTH)

        section("Local explanation", "What moves this prediction", "One-feature counterfactual sensitivity around this exact matchup.")
        impacts = []
        reference = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS)
        for feature in FEATURE_COLUMNS:
            perturbed = scenario_row.copy()
            perturbed[feature] = reference[feature].iloc[0]
            shifted = model.predict_proba(perturbed[FEATURE_COLUMNS])[0, 2]
            impacts.append({"feature": feature, "impact": probabilities[2] - shifted, "value": scenario_row[feature].iloc[0]})
        impact_frame = pd.DataFrame(impacts).assign(abs_impact=lambda frame: frame["impact"].abs()).nlargest(14, "abs_impact").sort_values("impact")
        fig = px.bar(
            impact_frame,
            x="impact",
            y="feature",
            orientation="h",
            color="impact",
            color_continuous_scale=[COLOR["red"], "#243850", COLOR["cyan"]],
            color_continuous_midpoint=0,
            title=f"Contribution to {home} win probability",
            hover_data={"value": ":.3f"},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat="+.1%")
        st.plotly_chart(style_chart(fig, 490), **FULL_WIDTH)

elif page == "Model Observatory":
    section(
        "Model risk",
        "Performance, calibration, ablation, and drift",
        "The interface exposes weak folds and changing inputs instead of compressing the model into one flattering score.",
    )
    metrics = st.columns(4)
    with metrics[0]:
        metric_card("Mean log loss", f"{mean_loss:.3f}", f"95% bootstrap CI {bootstrap['log_loss']['low']:.3f}–{bootstrap['log_loss']['high']:.3f}")
    with metrics[1]:
        metric_card("Mean RPS", f"{backtests['rps'].mean():.3f}", "ordinal probability error")
    with metrics[2]:
        metric_card("Calibration error", f"{backtests['ece'].mean():.3f}", "top-label expected calibration error")
    with metrics[3]:
        metric_card("Conformal efficiency", f"{conformal['single_outcome_rate']:.1%}", "matches resolved to one-outcome sets")

    section(
        "Latest challenger audit",
        "Production model versus staged architectures and data",
        "The observatory now includes every June 21 experiment, including candidates that failed.",
    )
    architecture_summary = (
        architectures.groupby("architecture", as_index=False)
        .agg(
            log_loss=("log_loss", "mean"),
            accuracy=("accuracy", "mean"),
            training_seconds=("training_seconds", "mean"),
        )
        .sort_values("log_loss")
    )
    challenger_view = advanced_matrix.loc[
        ~advanced_matrix["candidate"].isin(
            ["Baseline scores + state", "Calibrated hybrid"]
        )
    ].copy()
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            architecture_summary.sort_values("log_loss", ascending=False),
            x="log_loss",
            y="architecture",
            orientation="h",
            color="architecture",
            text="log_loss",
            color_discrete_map={
                "Calibrated hybrid": COLOR["cyan"],
                "CUDA XGBoost": COLOR["blue"],
                "CUDA residual network": COLOR["gold"],
            },
            title="Five-fold architecture benchmark",
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)
    with right:
        fig = px.scatter(
            challenger_view,
            x="log_loss_basis_points",
            y="accuracy_points",
            color="decision",
            symbol="stage",
            hover_name="candidate",
            color_discrete_map={
                "Promotion candidate": COLOR["cyan"],
                "Coverage-limited": COLOR["gold"],
                "Oracle only": COLOR["blue"],
                "Rejected": COLOR["red"],
            },
            title="All staged challengers",
            labels={
                "log_loss_basis_points": "Δ log loss · bp",
                "accuracy_points": "Δ accuracy · pp",
            },
        )
        fig.add_vline(x=0, line_color=COLOR["muted"])
        fig.add_hline(y=0, line_color=COLOR["muted"])
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backtests["edition"], y=backtests["baseline_log_loss"], mode="lines+markers", name="Historical prior", line={"color": COLOR["muted"], "dash": "dash"}))
        fig.add_trace(go.Scatter(x=backtests["edition"], y=backtests["log_loss"], mode="lines+markers", name="Calibrated ensemble", line={"color": COLOR["cyan"], "width": 3}))
        fig.update_layout(title="Walk-forward World Cup log loss", yaxis_title="Log loss")
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)
    with right:
        ladder = benchmarks.groupby("model", as_index=False).agg(log_loss=("log_loss", "mean"), ece=("ece", "mean"), accuracy=("accuracy", "mean")).sort_values("log_loss")
        fig = px.scatter(
            ladder,
            x="log_loss",
            y="ece",
            size="accuracy",
            color="model",
            text="model",
            title="Benchmark frontier: discrimination × calibration",
            labels={"log_loss": "Mean log loss", "ece": "Expected calibration error"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            reliability,
            x="predicted",
            y="observed",
            color="outcome",
            size="count",
            color_discrete_map={"Home win": COLOR["cyan"], "Draw": COLOR["gold"], "Away win": COLOR["red"]},
            title="Multiclass reliability surface",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": COLOR["muted"]})
        fig.update_xaxes(tickformat=".0%", range=[0, 0.9])
        fig.update_yaxes(tickformat=".0%", range=[0, 0.9])
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)
    with right:
        ablation_view = ablation.sort_values("log_loss_delta")
        fig = px.bar(
            ablation_view,
            x="log_loss_delta",
            y="configuration",
            orientation="h",
            color="log_loss_delta",
            color_continuous_scale=[COLOR["cyan"], "#263d55", COLOR["red"]],
            color_continuous_midpoint=0,
            title="Feature ablation on the unseen 2022 World Cup",
            labels={"log_loss_delta": "Log-loss change vs full feature system", "configuration": ""},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig), **FULL_WIDTH)

    left, right = st.columns([1.15, 1])
    with left:
        importance = pd.DataFrame(report["feature_importance"]).head(18).sort_values("importance")
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#28455f", COLOR["cyan"]],
            title="Held-out permutation importance",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig, 600), **FULL_WIDTH)
    with right:
        drift_view = drift.head(18).sort_values("psi")
        fig = px.bar(
            drift_view,
            x="psi",
            y="feature",
            orientation="h",
            color="status",
            color_discrete_map={"stable": COLOR["cyan"], "watch": COLOR["gold"], "material": COLOR["red"]},
            title="Population stability index · 2024+ vs historical",
        )
        fig.add_vline(x=0.1, line_dash="dot", line_color=COLOR["gold"])
        fig.add_vline(x=0.25, line_dash="dot", line_color=COLOR["red"])
        st.plotly_chart(style_chart(fig, 600), **FULL_WIDTH)

elif page == "Team Atlas":
    section(
        "Latent state",
        "A multidimensional map of the 2026 field",
        "Strength is shown with its uncertainty, trend, schedule quality, scoring profile and consistency, not as a single rank.",
    )
    active = ratings.loc[ratings["team"].isin(simulation["team"])].copy()
    certainty = 1 / active["rating_sigma"].clip(lower=1)
    active["certainty"] = (certainty - certainty.min()) / (certainty.max() - certainty.min())
    fig = px.scatter(
        active,
        x="attack",
        y="defence",
        size="certainty",
        color="elo",
        hover_name="team",
        hover_data={"rating_sigma": ":.1f", "momentum": ":+.3f", "schedule_strength": ":.0f", "consistency": ":.3f"},
        text="team",
        color_continuous_scale=[COLOR["red"], COLOR["gold"], COLOR["cyan"]],
        title="Attack–defence state space",
        labels={"attack": "Recency-weighted attacking output", "defence": "Recency-weighted goals conceded", "elo": "Power"},
    )
    fig.update_traces(textposition="top center")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_chart(fig, 650), **FULL_WIDTH)

    section(
        "Roster-aware atlas",
        "Team state beside official squad structure",
        "The production latent state remains result-driven; squad structure is displayed alongside it as a current 2026 intelligence layer.",
    )
    roster_atlas = squad_teams.merge(
        active[["team", "elo", "form", "attack", "defence"]],
        on="team",
        how="inner",
    )
    fig = px.scatter(
        roster_atlas,
        x="total_caps",
        y="total_international_goals",
        size="top_five_league_share",
        color="elo",
        hover_name="team",
        hover_data={
            "mean_age": ":.1f",
            "domestic_club_share": ":.0%",
            "club_diversity": ":.0%",
        },
        text="team",
        color_continuous_scale=[COLOR["red"], COLOR["gold"], COLOR["cyan"]],
        title="Official squad experience × scoring × model power",
        labels={
            "total_caps": "Combined international caps",
            "total_international_goals": "Combined international goals",
            "elo": "Dynamic power",
        },
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(style_chart(fig, 620), **FULL_WIDTH)

    selectors = st.columns([1, 1, 2])
    with selectors[0]:
        team_a = st.selectbox("Fingerprint A", active["team"].tolist(), index=0)
    with selectors[1]:
        team_b = st.selectbox("Fingerprint B", active["team"].tolist(), index=min(1, len(active) - 1))
    radar_features = ["form", "attack", "clean_sheet_rate", "consistency", "prestige", "schedule_strength"]
    normalized = active.set_index("team")[radar_features].copy()
    for column in radar_features:
        low, high = normalized[column].min(), normalized[column].max()
        normalized[column] = (normalized[column] - low) / max(high - low, 1e-9)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            embedding,
            x="embedding_x",
            y="embedding_y",
            color="archetype",
            text="team",
            hover_name="team",
            title="Latent team-state embedding · PCA + archetype clustering",
            labels={"embedding_x": "Latent dimension 1", "embedding_y": "Latent dimension 2"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(style_chart(fig, 530), **FULL_WIDTH)
    with right:
        fig = go.Figure()
        for team, color in ((team_a, COLOR["cyan"]), (team_b, COLOR["gold"])):
            values = normalized.loc[team].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    name=team,
                    line={"color": color},
                )
            )
        fig.update_layout(
            title="Team fingerprint comparison",
            polar={"radialaxis": {"visible": True, "range": [0, 1], "gridcolor": COLOR["line"]}},
        )
        st.plotly_chart(style_chart(fig, 530), **FULL_WIDTH)

    st.dataframe(
        active[
            [
                "team", "elo", "rating_sigma", "power_low", "power_high", "form", "momentum",
                "attack", "defence", "clean_sheet_rate", "consistency", "schedule_strength",
            ]
        ].sort_values("elo", ascending=False),
        **FULL_WIDTH,
        hide_index=True,
        column_config={
            "elo": st.column_config.NumberColumn("Power", format="%.0f"),
            "rating_sigma": st.column_config.NumberColumn("Uncertainty", format="%.1f"),
            "form": st.column_config.ProgressColumn("Form", min_value=0, max_value=1, format="%.2f"),
            "clean_sheet_rate": st.column_config.ProgressColumn("Clean sheet", min_value=0, max_value=1, format="%.0%%"),
            "consistency": st.column_config.ProgressColumn("Consistency", min_value=0, max_value=1, format="%.2f"),
        },
    )

else:
    section(
        "Reproducible evidence",
        "Research artifacts and publication graphics",
        "Every visual below is generated deterministically from the same versioned experiment outputs used by the app.",
    )
    visuals = [
        ("Cinematic hero", VISUAL_DIR / "hero-data-field.png"),
        ("Türkiye focus", VISUAL_DIR / "turkiye-focus.png"),
        ("Türkiye blind audit", VISUAL_DIR / "turkiye-blind-forecast.png"),
        ("Blind tournament audit", VISUAL_DIR / "blind-tournament-audit.png"),
        ("Model evolution", VISUAL_DIR / "model-evolution.png"),
        ("Advanced research", VISUAL_DIR / "advanced-research.png"),
        ("Official squad intelligence", VISUAL_DIR / "squad-intelligence.png"),
        ("Model evidence", VISUAL_DIR / "model-evidence.png"),
        ("Tournament forecast", VISUAL_DIR / "tournament-forecast.png"),
        ("Team power map", VISUAL_DIR / "team-power-map.png"),
        ("Social preview", VISUAL_DIR / "forecast-social-card.png"),
    ]
    for index in range(0, len(visuals), 2):
        columns = st.columns(2)
        for column, (title, path) in zip(columns, visuals[index : index + 2]):
            with column:
                st.markdown(f"#### {title}")
                st.image(path, **FULL_WIDTH)
                st.download_button(
                    f"Download {title}",
                    path.read_bytes(),
                    file_name=path.name,
                    mime="image/png",
                    key=path.name,
                )

    section("Lineage", "Experiment manifest", "Enough provenance to reproduce and challenge the result.")
    manifest = pd.DataFrame(
        [
            ["Experiment", report["experiment"]["id"]],
            ["Dataset SHA-256", report["experiment"]["data_sha256"]],
            ["Python", report["experiment"]["python"]],
            ["scikit-learn", report["experiment"]["scikit_learn"]],
            ["Training window", f"{report['experiment']['training_window_years']} years"],
            ["Random seed", str(report["experiment"]["random_seed"])],
            ["Simulation runs", f"{report['experiment']['simulation_runs']:,}"],
            ["Advanced report", advanced_report["generated_at"]],
            ["CUDA device", advanced_report["runtime"]["cuda_device"]],
            ["Advanced candidates promoted", str(promoted_candidates)],
            [
                "Official squad join",
                f"{advanced_report['coverage']['official_squad_join_rate']:.1%}",
            ],
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(manifest, **FULL_WIDTH, hide_index=True)

    st.markdown(
        f"""
        <div class="evidence">
        <b>Interpretation boundary.</b> This system forecasts distributions, not certainties.
        Production probabilities do not observe injuries, tactical assignments, authenticated club
        workload, or market prices. Event-level xG, official squads, environment, and announced-lineup
        features are now implemented as staged layers, but none passed the complete production gate.
        The internal score process is retained only as a diagnostic; public forecasts are presented
        as win, draw, and loss probabilities. Current empirical 90% conformal coverage is
        <b>{conformal['empirical_coverage']:.1%}</b>; calibration and drift remain visible because
        hiding model risk would make the product less technically credible.
        </div>
        """,
        unsafe_allow_html=True,
    )


