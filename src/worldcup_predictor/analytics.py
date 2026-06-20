from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TEAM_EMBEDDING_FEATURES = [
    "elo",
    "form",
    "momentum",
    "attack",
    "defence",
    "attack_trend",
    "defence_trend",
    "clean_sheet_rate",
    "scoring_rate",
    "consistency",
    "schedule_strength",
    "prestige",
    "volatility",
]


def team_state_embedding(
    ratings: pd.DataFrame, tournament_teams: list[str], clusters: int = 6
) -> pd.DataFrame:
    """Project the current multidimensional team state into a reproducible 2D atlas."""
    frame = ratings.loc[ratings["team"].isin(tournament_teams)].copy()
    matrix = StandardScaler().fit_transform(frame[TEAM_EMBEDDING_FEATURES])
    embedding = PCA(n_components=2, random_state=2026).fit_transform(matrix)
    labels = KMeans(n_clusters=clusters, random_state=2026, n_init=20).fit_predict(matrix)
    frame["embedding_x"] = embedding[:, 0]
    frame["embedding_y"] = embedding[:, 1]
    frame["cluster"] = labels
    cluster_profiles = (
        frame.groupby("cluster")[["attack", "defence", "form", "volatility"]].mean()
    )
    attack_rank = cluster_profiles["attack"].rank(pct=True)
    defence_rank = (-cluster_profiles["defence"]).rank(pct=True)
    form_rank = cluster_profiles["form"].rank(pct=True)
    volatility_rank = (-cluster_profiles["volatility"]).rank(pct=True)
    archetypes = {}
    for cluster in cluster_profiles.index:
        scores = {
            "High press / attack": attack_rank[cluster],
            "Control / defence": defence_rank[cluster],
            "In-form contenders": form_rank[cluster],
            "Stable operators": volatility_rank[cluster],
        }
        archetypes[cluster] = max(scores, key=scores.get)
    frame["archetype"] = frame["cluster"].map(archetypes)
    return frame[
        ["team", "embedding_x", "embedding_y", "cluster", "archetype"]
    ].sort_values("team")

