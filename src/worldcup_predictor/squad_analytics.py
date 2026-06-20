from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL_SQUAD_PDF = (
    "https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf"
)
TOP_LEAGUE_CODES = {"ENG", "ESP", "GER", "ITA", "FRA"}


def squad_analytics_2026(
    squads: list[dict[str, object]],
    official_pdf_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official = (
        parse_official_squad_pdf(official_pdf_path)
        if official_pdf_path and official_pdf_path.exists()
        else pd.DataFrame()
    )
    players = []
    tournament_date = pd.Timestamp("2026-06-11")
    for squad in squads:
        for player in squad["players"]:
            birth_date = pd.Timestamp(player["date_of_birth"])
            club_country = player.get("club", {}).get("country")
            players.append(
                {
                    "team": squad["name"],
                    "fifa_code": squad["fifa_code"],
                    "group": squad["group"],
                    "number": player["number"],
                    "position": player["pos"],
                    "player": player["name"],
                    "birth_date": birth_date,
                    "age": (tournament_date - birth_date).days / 365.25,
                    "club": player.get("club", {}).get("name"),
                    "club_country": club_country,
                    "top_five_league": club_country in TOP_LEAGUE_CODES,
                    "domestic_club": club_country == squad["fifa_code"],
                }
            )
    player_frame = pd.DataFrame(players)
    if not official.empty:
        player_frame = player_frame.merge(
            official[
                [
                    "fifa_code",
                    "number",
                    "height",
                    "caps",
                    "international_goals",
                ]
            ],
            on=["fifa_code", "number"],
            how="left",
        )
    else:
        player_frame[["height", "caps", "international_goals"]] = np.nan
    rows = []
    for (team, code, group), frame in player_frame.groupby(
        ["team", "fifa_code", "group"]
    ):
        forward = frame.loc[frame["position"].eq("FW")]
        caps = frame["caps"].fillna(0)
        goals = frame["international_goals"].fillna(0)
        rows.append(
            {
                "team": team,
                "fifa_code": code,
                "group": group,
                "players": len(frame),
                "mean_age": frame["age"].mean(),
                "age_std": frame["age"].std(),
                "under_23_share": frame["age"].lt(23).mean(),
                "over_30_share": frame["age"].ge(30).mean(),
                "mean_height": frame["height"].mean(),
                "total_caps": caps.sum(),
                "mean_caps": caps.mean(),
                "total_international_goals": goals.sum(),
                "top3_cap_share": caps.nlargest(3).sum() / max(caps.sum(), 1),
                "top3_goal_share": goals.nlargest(3).sum() / max(goals.sum(), 1),
                "forward_goal_depth": (
                    forward["international_goals"].fillna(0).sum()
                    / max(len(forward), 1)
                ),
                "top_five_league_share": frame["top_five_league"].mean(),
                "domestic_club_share": frame["domestic_club"].mean(),
                "club_diversity": frame["club"].nunique() / max(len(frame), 1),
                "goalkeeper_share": frame["position"].eq("GK").mean(),
                "defender_share": frame["position"].eq("DF").mean(),
                "midfielder_share": frame["position"].eq("MF").mean(),
                "forward_share": frame["position"].eq("FW").mean(),
                "official_stats_coverage": frame["caps"].notna().mean(),
            }
        )
    return player_frame, pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def parse_official_squad_pdf(path: Path) -> pd.DataFrame:
    from pypdf import PdfReader

    rows = []
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout")
        team_match = re.search(r"^\s*(.+?) \(([A-Z]{3})\)\s*$", text, re.M)
        if not team_match:
            continue
        team, code = team_match.groups()
        for line in text.splitlines():
            player_match = re.match(
                r"^\s*(\d+)\s+(GK|DF|MF|FW)\s+(.+?)\s+"
                r"(\d{2}/\d{2}/\d{4})(.+?)\s+(\d{3})\s+(\d+)\s+(\d+)\s*$",
                line,
            )
            if not player_match:
                continue
            (
                number,
                position,
                _,
                birth_date,
                club,
                height,
                caps,
                goals,
            ) = player_match.groups()
            rows.append(
                {
                    "team": team,
                    "fifa_code": code,
                    "number": int(number),
                    "position": position,
                    "birth_date": pd.to_datetime(
                        birth_date, format="%d/%m/%Y"
                    ),
                    "club_official": club.strip(),
                    "height": int(height),
                    "caps": int(caps),
                    "international_goals": int(goals),
                }
            )
    return pd.DataFrame(rows)
