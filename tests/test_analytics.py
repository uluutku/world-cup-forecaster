import pandas as pd

from worldcup_predictor.analytics import TEAM_EMBEDDING_FEATURES, team_state_embedding


def test_team_embedding_is_deterministic_and_complete():
    rows = []
    for index in range(12):
        row = {"team": f"T{index}"}
        row.update({feature: index + offset / 10 for offset, feature in enumerate(TEAM_EMBEDDING_FEATURES)})
        rows.append(row)
    ratings = pd.DataFrame(rows)
    first = team_state_embedding(ratings, ratings["team"].tolist(), clusters=3)
    second = team_state_embedding(ratings, ratings["team"].tolist(), clusters=3)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first["archetype"].notna().all()
