from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import CONDITIONS_PATH, RAW_DIR

WEATHER_FEATURE_COLUMNS = [
    "weather_temperature",
    "weather_apparent_temperature",
    "weather_humidity",
    "weather_precipitation",
    "weather_wind",
    "weather_gust",
    "weather_pressure",
    "weather_elevation",
    "weather_heat_stress",
    "weather_cold_stress",
    "weather_missing",
]

TRAVEL_FEATURE_COLUMNS = [
    "travel_distance_diff",
    "travel_distance_mean",
    "timezone_shift_diff",
    "travel_missing",
]

CONDITION_FEATURE_COLUMNS = WEATHER_FEATURE_COLUMNS + TRAVEL_FEATURE_COLUMNS

TEAM_COUNTRY_ALIASES = {
    "United States": "United States of America",
    "South Korea": "South Korea",
    "North Korea": "North Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "DR Congo": "Democratic Republic of the Congo",
    "Russia": "Russia",
    "Iran": "Iran",
    "Turkey": "Türkiye",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "Chinese Taipei": "Taiwan",
    "Hong Kong": "Hong Kong",
    "Palestine": "Palestine",
}


def build_world_cup_conditions(
    reference_matches: pd.DataFrame,
    refresh: bool = False,
    workers: int = 8,
) -> pd.DataFrame:
    """Fetch and cache World Cup venue, ERA5 weather, altitude, and travel context."""
    if CONDITIONS_PATH.exists() and not refresh:
        return pd.read_parquet(CONDITIONS_PATH)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches = reference_matches.loc[
        reference_matches["tournament_name"].str.contains(
            "Men's World Cup", na=False
        )
        & reference_matches["match_date"].dt.year.ge(1950)
    ].copy()
    cities = (
        matches[["city_name", "country_name"]]
        .drop_duplicates()
        .sort_values(["country_name", "city_name"])
    )
    geo_cache_path = RAW_DIR / "geocoded_world_cup_cities.json"
    geo_cache = (
        json.loads(geo_cache_path.read_text(encoding="utf-8"))
        if geo_cache_path.exists()
        else {}
    )
    for row in cities.itertuples(index=False):
        key = f"{row.city_name}|{row.country_name}"
        if key not in geo_cache:
            geo_cache[key] = _geocode_city(row.city_name, row.country_name)
            time.sleep(0.08)
    geo_cache_path.write_text(
        json.dumps(geo_cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    capitals = _country_capitals(refresh=refresh)
    weather_cache_path = RAW_DIR / "world_cup_weather_cache.json"
    weather_cache = (
        json.loads(weather_cache_path.read_text(encoding="utf-8"))
        if weather_cache_path.exists()
        else {}
    )
    jobs = {}
    for row in matches.itertuples():
        geo = geo_cache.get(f"{row.city_name}|{row.country_name}")
        if not geo:
            continue
        key = f"{row.match_date.date()}|{geo['latitude']:.4f}|{geo['longitude']:.4f}"
        if key not in weather_cache:
            jobs[key] = (
                row.match_date.date().isoformat(),
                geo["latitude"],
                geo["longitude"],
            )
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_weather, *values): key
                for key, values in jobs.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                try:
                    weather_cache[key] = future.result()
                except Exception as error:
                    weather_cache[key] = {"error": str(error)}
                if completed % 50 == 0:
                    weather_cache_path.write_text(
                        json.dumps(weather_cache), encoding="utf-8"
                    )
        weather_cache_path.write_text(
            json.dumps(weather_cache), encoding="utf-8"
        )
    rows = []
    for match in matches.itertuples():
        geo = geo_cache.get(f"{match.city_name}|{match.country_name}")
        weather = {}
        if geo:
            key = (
                f"{match.match_date.date()}|{geo['latitude']:.4f}|"
                f"{geo['longitude']:.4f}"
            )
            weather = weather_cache.get(key, {})
        hour = _match_hour(match.match_time)
        hourly = _hourly_values(weather, hour)
        home_capital = _capital_for_team(match.home_team_name, capitals)
        away_capital = _capital_for_team(match.away_team_name, capitals)
        home_distance = _distance(home_capital, geo)
        away_distance = _distance(away_capital, geo)
        venue_offset = _timezone_offset(geo.get("timezone")) if geo else None
        home_offset = home_capital.get("utc_offset") if home_capital else None
        away_offset = away_capital.get("utc_offset") if away_capital else None
        rows.append(
            {
                "date": match.match_date,
                "home_team": match.home_team_name,
                "away_team": match.away_team_name,
                "city": match.city_name,
                "weather_temperature": hourly.get("temperature_2m"),
                "weather_apparent_temperature": hourly.get(
                    "apparent_temperature"
                ),
                "weather_humidity": hourly.get("relative_humidity_2m"),
                "weather_precipitation": hourly.get("precipitation"),
                "weather_wind": hourly.get("wind_speed_10m"),
                "weather_gust": hourly.get("wind_gusts_10m"),
                "weather_pressure": hourly.get("surface_pressure"),
                "weather_elevation": geo.get("elevation") if geo else None,
                "travel_home_km": home_distance,
                "travel_away_km": away_distance,
                "timezone_home_shift": (
                    abs(home_offset - venue_offset)
                    if home_offset is not None and venue_offset is not None
                    else None
                ),
                "timezone_away_shift": (
                    abs(away_offset - venue_offset)
                    if away_offset is not None and venue_offset is not None
                    else None
                ),
            }
        )
    frame = pd.DataFrame(rows)
    CONDITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CONDITIONS_PATH, index=False)
    return frame


def add_condition_features(
    features: pd.DataFrame, conditions: pd.DataFrame
) -> pd.DataFrame:
    frame = features.merge(
        conditions,
        on=["date", "home_team", "away_team"],
        how="left",
    )
    raw_columns = [
        "weather_temperature",
        "weather_apparent_temperature",
        "weather_humidity",
        "weather_precipitation",
        "weather_wind",
        "weather_gust",
        "weather_pressure",
        "weather_elevation",
        "travel_home_km",
        "travel_away_km",
        "timezone_home_shift",
        "timezone_away_shift",
    ]
    for column in raw_columns:
        if column not in frame:
            frame[column] = np.nan
    frame["weather_heat_stress"] = np.clip(
        frame["weather_apparent_temperature"] - 26.0, 0, None
    )
    frame["weather_cold_stress"] = np.clip(
        8.0 - frame["weather_apparent_temperature"], 0, None
    )
    frame["weather_missing"] = frame["weather_temperature"].isna().astype(float)
    frame["travel_distance_diff"] = (
        frame["travel_home_km"] - frame["travel_away_km"]
    ) / 5000.0
    frame["travel_distance_mean"] = (
        frame["travel_home_km"] + frame["travel_away_km"]
    ) / 10000.0
    frame["timezone_shift_diff"] = (
        frame["timezone_home_shift"] - frame["timezone_away_shift"]
    ) / 6.0
    frame["travel_missing"] = (
        frame[["travel_home_km", "travel_away_km"]].isna().any(axis=1).astype(float)
    )
    defaults = {
        "weather_temperature": 18.0,
        "weather_apparent_temperature": 18.0,
        "weather_humidity": 60.0,
        "weather_precipitation": 0.0,
        "weather_wind": 8.0,
        "weather_gust": 15.0,
        "weather_pressure": 1013.0,
        "weather_elevation": 0.0,
        "weather_heat_stress": 0.0,
        "weather_cold_stress": 0.0,
        "travel_distance_diff": 0.0,
        "travel_distance_mean": 0.0,
        "timezone_shift_diff": 0.0,
    }
    for column, default in defaults.items():
        frame[column] = frame[column].fillna(default)
    frame["weather_temperature"] = (
        frame["weather_temperature"] - 18.0
    ) / 12.0
    frame["weather_apparent_temperature"] = (
        frame["weather_apparent_temperature"] - 18.0
    ) / 12.0
    frame["weather_humidity"] = (frame["weather_humidity"] - 60.0) / 30.0
    frame["weather_precipitation"] = np.log1p(
        frame["weather_precipitation"].clip(lower=0)
    )
    frame["weather_wind"] = frame["weather_wind"] / 25.0
    frame["weather_gust"] = frame["weather_gust"] / 40.0
    frame["weather_pressure"] = (frame["weather_pressure"] - 1013.0) / 30.0
    frame["weather_elevation"] = frame["weather_elevation"] / 2000.0
    frame["weather_heat_stress"] = frame["weather_heat_stress"] / 10.0
    frame["weather_cold_stress"] = frame["weather_cold_stress"] / 10.0
    return frame


def _geocode_city(city: str, country: str) -> dict[str, object] | None:
    query = urlencode(
        {"name": city, "count": 10, "language": "en", "format": "json"}
    )
    payload = _get_json(
        f"https://geocoding-api.open-meteo.com/v1/search?{query}"
    )
    results = payload.get("results", [])
    if not results:
        return None
    country_lower = str(country).lower()
    selected = next(
        (
            result
            for result in results
            if country_lower in str(result.get("country", "")).lower()
            or str(result.get("country", "")).lower() in country_lower
        ),
        results[0],
    )
    return {
        "latitude": selected["latitude"],
        "longitude": selected["longitude"],
        "elevation": selected.get("elevation", 0),
        "timezone": selected.get("timezone"),
        "country": selected.get("country"),
    }


def _fetch_weather(date: str, latitude: float, longitude: float) -> dict:
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "wind_speed_10m",
                "wind_gusts_10m",
                "surface_pressure",
            ]
        ),
        "timezone": "auto",
        "models": "era5",
    }
    return _get_json(
        f"https://archive-api.open-meteo.com/v1/archive?{urlencode(parameters)}",
        retries=4,
    )


def _get_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "WorldCupIntelligence/2.0"})
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def _hourly_values(payload: dict, hour: int) -> dict[str, float]:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return {}
    index = min(max(hour, 0), len(hourly["time"]) - 1)
    return {
        key: values[index]
        for key, values in hourly.items()
        if key != "time" and isinstance(values, list) and len(values) > index
    }


def _match_hour(value: object) -> int:
    if pd.isna(value):
        return 15
    try:
        return int(str(value).split(":")[0])
    except ValueError:
        return 15


def _country_capitals(refresh: bool = False) -> dict[str, dict[str, object]]:
    path = RAW_DIR / "country_capitals.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    payload = _get_json(
        "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"
    )
    output = {}
    for country in payload:
        latlng = country.get("latlng")
        if not latlng:
            continue
        offset = float(np.clip(round(latlng[1] / 15), -12, 14))
        record = {
            "latitude": latlng[0],
            "longitude": latlng[1],
            "utc_offset": offset,
        }
        names = {
            country.get("name", {}).get("common"),
            country.get("name", {}).get("official"),
        }
        for name in names:
            if name:
                output[name] = record
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def _capital_for_team(
    team: str, capitals: dict[str, dict[str, object]]
) -> dict[str, object] | None:
    name = TEAM_COUNTRY_ALIASES.get(team, team)
    if name in capitals:
        return capitals[name]
    return next(
        (
            value
            for country, value in capitals.items()
            if name.lower() in country.lower() or country.lower() in name.lower()
        ),
        None,
    )


def _distance(
    first: dict[str, object] | None, second: dict[str, object] | None
) -> float | None:
    if not first or not second:
        return None
    lat1, lon1 = math.radians(first["latitude"]), math.radians(first["longitude"])
    lat2, lon2 = math.radians(second["latitude"]), math.radians(second["longitude"])
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(value))


def _parse_offset(value: str | None) -> float | None:
    if not value or not value.startswith("UTC"):
        return None
    if value == "UTC":
        return 0.0
    sign = -1 if "-" in value else 1
    text = value.replace("UTC+", "").replace("UTC-", "")
    hours, minutes = (text.split(":") + ["0"])[:2]
    return sign * (float(hours) + float(minutes) / 60)


def _timezone_offset(timezone_name: str | None) -> float | None:
    if not timezone_name:
        return None
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        offset = datetime.now(ZoneInfo(timezone_name)).utcoffset()
        return offset.total_seconds() / 3600 if offset else 0.0
    except Exception:
        return None
