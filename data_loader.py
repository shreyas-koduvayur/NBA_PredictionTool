"""
data_loader.py
==============
Fetches NBA player stats using nba_api, with a patched session that adds
the two headers stats.nba.com requires but nba_api's defaults omit.

Why it was failing before
--------------------------
nba_api ships with good default headers (STATS_HEADERS) but is missing:
  - x-nba-stats-origin: stats
  - x-nba-stats-token:  true
stats.nba.com checks for both of these and drops the connection without them.

The fix is simple: grab nba_api's shared requests.Session via
NBAStatsHTTP.set_session(), add the two missing headers, then make calls
as normal. nba_api reuses that same session for every request.
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.library.http import NBAStatsHTTP, STATS_HEADERS

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 3600


def _cache_path(key: str) -> Path:
    hashed = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{hashed}.json"


def _load_from_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def _save_to_cache(key: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_cache_path(key), "w") as f:
        json.dump(df.to_dict("records"), f)


# ── Session setup ──────────────────────────────────────────────────────────────
def _configure_session() -> None:
    """
    Inject a pre-warmed requests.Session into nba_api's HTTP client.

    nba_api's STATS_HEADERS already has a good User-Agent, Referer, etc.
    We extend it with the two tokens stats.nba.com additionally requires,
    then visit nba.com to pick up any cookies the CDN checks for.
    """
    session = requests.Session()

    full_headers = {
        **STATS_HEADERS,
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token":  "true",
    }
    session.headers.update(full_headers)

    # Warm up cookies — visit the main site before hitting the stats subdomain
    try:
        session.get("https://www.nba.com", timeout=10)
        time.sleep(1)
    except Exception:
        pass  # If nba.com is unreachable we'll find out on the stats call

    # Hand the session to nba_api — it will use it for all subsequent requests
    NBAStatsHTTP.set_session(session)


# ── Fetch helpers ──────────────────────────────────────────────────────────────
def _fetch_base(season: str) -> pd.DataFrame:
    """Per-game box score stats from leaguedashplayerstats Base."""
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        timeout=30,
    )
    return resp.get_data_frames()[0]


def _fetch_advanced(season: str) -> pd.DataFrame:
    """
    Advanced stats from leaguedashplayerstats Advanced measure type.
    Expected columns: OFF_RATING, DEF_RATING, NET_RATING, PIE, TS_PCT, USG_PCT, PACE.

    PIE (Player Impact Estimate) is the NBA's official efficiency metric:
      (PTS + FGM + FTM - FGA - FTA + DREB + 0.5*OREB + AST + STL + 0.5*BLK - PF - TO)
      divided by the same sum for all players in the game.
    This is far more accurate than our hand-computed PER proxy.
    """
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=30,
    )
    return resp.get_data_frames()[0]


# ── Public API ─────────────────────────────────────────────────────────────────
def load_player_stats(season: str = "2024-25") -> tuple[pd.DataFrame, bool, Optional[str]]:
    """
    Load player stats for the given season from stats.nba.com.
    Merges Base stats + Bio stats for NET_RATING.

    Returns (df, is_mock, error_message).
    """
    cache_key = f"nba_stats_{season}"
    cached = _load_from_cache(cache_key)
    if cached is not None:
        return cached, False, None

    try:
        _configure_session()

        df = _fetch_base(season)

        # ── Fetch Advanced stats — retry once if first attempt fails ─────────
        df_adv = None
        for attempt in range(2):
            try:
                time.sleep(2 + attempt)  # wait longer on retry
                df_adv = _fetch_advanced(season)
                print(f"[data_loader] Advanced cols: {list(df_adv.columns)}")
                break
            except Exception as adv_exc:
                print(f"[data_loader] Advanced fetch attempt {attempt+1} failed: {adv_exc}")
                if attempt == 0:
                    # Re-warm the session before retrying
                    _configure_session()

        if df_adv is not None:
            adv_grab = ["PLAYER_ID"]
            for col in ["PIE", "NET_RATING", "OFF_RATING", "DEF_RATING", "TS_PCT", "USG_PCT"]:
                if col in df_adv.columns:
                    adv_grab.append(col)
            df = pd.merge(df, df_adv[adv_grab], on="PLAYER_ID", how="left")
            print(f"[data_loader] Merged advanced cols: {adv_grab}")
        else:
            print("[data_loader] Advanced fetch failed after retries — using proxy PER")
            df["PIE"] = None
            df["NET_RATING"] = 0.0

        # ── PER: use PIE if available, else NET_RATING blend, else pure proxy ──
        if "PIE" in df.columns and df["PIE"].notna().any():
            df["PER"] = df["PIE"] * 150
            print("[data_loader] ✓ Using real PIE for PER")
        elif "NET_RATING" in df.columns and df["NET_RATING"].notna().any() and (df["NET_RATING"] != 0).any():
            # NET_RATING from Advanced — much better than pure box score proxy
            # Blend: scoring efficiency from box score + team impact from NET_RATING
            per_raw = (
                df["PTS"]
                + (df["REB"] * 0.2)
                + (df["AST"] * 0.7)
                + (df["STL"] * 0.5)
                + (df["BLK"] * 0.3)
                - (df["FGA"] - df["FGM"])
                - (df["FTA"] - df["FTM"])
                - df["TOV"]
            )
            df["PER"] = per_raw * 0.5 + df["NET_RATING"] * 1.5
            print("[data_loader] ✓ Using NET_RATING blend for PER")
        else:
            # Pure proxy — least accurate, but still functional
            per_raw = (
                df["PTS"]
                + (df["REB"] * 0.2)
                + (df["AST"] * 0.7)
                + (df["STL"] * 0.5)
                + (df["BLK"] * 0.3)
                - (df["FGA"] - df["FGM"])
                - (df["FTA"] - df["FTM"])
                - df["TOV"]
            )
            df["PER"] = per_raw
            print("[data_loader] ⚠ Using proxy PER — Advanced endpoint unavailable")

        df["PER_PER_MIN"] = df["PER"] / df["MIN"].replace(0, 1)

        # ── NET_RATING fallback ────────────────────────────────────────────────
        if "NET_RATING" not in df.columns:
            df["NET_RATING"] = 0.0
        df["NET_RATING"] = df["NET_RATING"].fillna(0)

        # W_PCT and WS48
        # WS48 uses NET_RATING (team impact) not PIE (individual efficiency)
        # This makes WS48 and PER_PER_MIN measure DIFFERENT things:
        #   PER_PER_MIN = how efficient the player is individually (PIE)
        #   WS48 = how much the team wins when the player is on court (NET_RATING)
        # Previously both used PIE — Giannis was getting double-counted
        total_games = (df["W"] + df["L"]).replace(0, 1)
        df["W_PCT"] = df["W"] / total_games
        net = df["NET_RATING"].fillna(0)
        # Scale NET_RATING (typically -5 to +15) to a WS48-like range
        # Baseline WS48 of 0.1 + net rating contribution
        df["WS48"] = 0.100 + (net / 100)

        # TS% from base stats
        fga_fta = (2 * (df["FGA"] + 0.44 * df["FTA"])).replace(0, 1)
        df["TS_PCT"] = df["PTS"] / fga_fta

        # NET_RATING already merged — keep it as a standalone column too
        df["USG_PCT"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / (df["GP"] * 20)

        df = df.fillna(0)
        _save_to_cache(cache_key, df)
        return df, False, None

    except Exception as exc:
        error_msg = str(exc)
        short = error_msg[:400] if len(error_msg) > 400 else error_msg
        print(f"[data_loader] NBA API error: {short}\nFalling back to demo data.")
        return _get_mock_data(), True, short


# ── Demo / fallback data ───────────────────────────────────────────────────────
def _get_mock_data() -> pd.DataFrame:
    """
    Approximate 2024-25 stats for ~25 players.
    These are ESTIMATES, not live numbers. The app shows DEMO DATA when active.
    """
    players = [
        # name, team, gp, pts, ast, reb, stl, blk, fg%, 3p%, ft%, ts%, usg%, min, net_rating, pie
        ("Shai Gilgeous-Alexander", "OKC", 73, 32.7,  6.4, 5.1,  2.0, 1.0, 0.535, 0.355, 0.897, 0.641, 30.8, 34.2, 12.1, 0.185),
        ("Nikola Jokic",            "DEN", 76, 29.6, 10.2, 12.7, 1.7, 0.7, 0.578, 0.359, 0.820, 0.660, 27.5, 34.9, 10.8, 0.210),
        ("Giannis Antetokounmpo",   "MIL", 74, 30.4,  6.5, 11.9, 1.2, 1.1, 0.611, 0.274, 0.717, 0.640, 33.1, 33.5,  7.2, 0.188),
        ("LeBron James",            "LAL", 71, 23.7,  8.3,  8.0, 1.2, 0.6, 0.540, 0.410, 0.750, 0.617, 25.6, 34.7,  4.1, 0.152),
        ("Jayson Tatum",            "BOS", 76, 26.9,  5.0,  8.1, 1.1, 0.6, 0.471, 0.381, 0.830, 0.601, 28.4, 35.4,  9.8, 0.158),
        ("Luka Doncic",             "DAL", 70, 28.1,  8.7,  8.6, 1.4, 0.5, 0.476, 0.376, 0.796, 0.612, 36.1, 35.8,  6.4, 0.172),
        ("Joel Embiid",             "PHI", 39, 34.7,  5.6, 11.0, 1.2, 1.7, 0.528, 0.342, 0.875, 0.654, 36.5, 33.9, 11.2, 0.195),
        ("Anthony Davis",           "LAL", 76, 25.7,  3.5, 12.6, 1.2, 2.3, 0.566, 0.262, 0.798, 0.626, 27.0, 34.1,  5.9, 0.162),
        ("Donovan Mitchell",        "CLE", 72, 26.6,  6.1,  5.1, 1.5, 0.4, 0.488, 0.381, 0.854, 0.610, 29.7, 33.8,  8.3, 0.158),
        ("Cade Cunningham",         "DET", 75, 25.5,  9.0,  4.4, 1.4, 0.5, 0.450, 0.330, 0.842, 0.572, 29.3, 34.5,  2.1, 0.148),
        ("Tyrese Haliburton",       "IND", 69, 20.1, 10.9,  4.3, 1.5, 0.5, 0.469, 0.405, 0.855, 0.598, 24.2, 32.8,  5.5, 0.148),
        ("Devin Booker",            "PHX", 68, 25.7,  6.5,  4.5, 1.1, 0.3, 0.490, 0.375, 0.862, 0.609, 28.1, 34.0, -1.2, 0.152),
        ("Karl-Anthony Towns",      "NYK", 74, 24.2,  3.3, 13.9, 0.9, 1.4, 0.529, 0.401, 0.842, 0.635, 24.6, 32.8,  6.1, 0.145),
        ("Bam Adebayo",             "MIA", 71, 19.3,  3.6, 10.4, 1.3, 0.9, 0.541, 0.182, 0.775, 0.600, 22.0, 33.7,  3.2, 0.138),
        ("Jalen Brunson",           "NYK", 77, 25.9,  7.4,  3.6, 0.9, 0.2, 0.482, 0.416, 0.874, 0.621, 30.8, 33.9,  7.4, 0.152),
        ("De'Aaron Fox",            "SAC", 74, 25.8,  6.1,  4.7, 1.5, 0.5, 0.490, 0.341, 0.773, 0.590, 28.5, 33.2, -0.8, 0.148),
        ("Evan Mobley",             "CLE", 76, 18.5,  3.0,  9.4, 1.3, 1.8, 0.560, 0.350, 0.740, 0.607, 19.5, 32.6,  7.9, 0.138),
        ("Victor Wembanyama",       "SAS", 71, 24.6,  3.9, 10.6, 1.2, 3.6, 0.484, 0.320, 0.793, 0.607, 25.4, 29.9,  4.8, 0.170),
        ("Alperen Sengun",          "HOU", 75, 21.1,  5.8,  9.3, 1.2, 1.7, 0.559, 0.338, 0.741, 0.598, 24.2, 31.4,  5.1, 0.148),
        ("Paolo Banchero",          "ORL", 80, 25.6,  5.8,  7.8, 0.9, 0.5, 0.482, 0.327, 0.758, 0.578, 29.2, 33.7,  3.7, 0.148),
        ("Trae Young",              "ATL", 77, 23.8, 10.8,  3.3, 1.3, 0.2, 0.434, 0.355, 0.907, 0.586, 31.4, 34.3, -4.1, 0.138),
        ("Darius Garland",          "CLE", 70, 21.7,  7.8,  2.7, 1.4, 0.2, 0.474, 0.388, 0.859, 0.598, 24.7, 32.2,  6.8, 0.138),
        ("Scottie Barnes",          "TOR", 74, 20.0,  6.1,  8.2, 1.5, 0.9, 0.502, 0.339, 0.753, 0.578, 23.3, 34.0, -2.3, 0.132),
        ("Jalen Williams",          "OKC", 74, 23.0,  5.8,  4.5, 1.3, 0.5, 0.507, 0.360, 0.835, 0.610, 26.1, 33.1,  8.9, 0.148),
        ("Domantas Sabonis",        "SAC", 75, 20.2,  8.2, 13.7, 1.2, 0.4, 0.575, 0.200, 0.692, 0.610, 22.8, 32.5, -0.3, 0.138),
    ]
    columns = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "GP",
        "PTS", "AST", "REB", "STL", "BLK",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "TS_PCT", "USG_PCT", "MIN", "NET_RATING", "PIE",
    ]
    df = pd.DataFrame(players, columns=columns)
    df["PLAYER_ID"] = range(1, len(df) + 1)

    # Use PIE directly — scaled to PER-like range (* 150)
    df["PER"] = df["PIE"] * 150
    df["PER_PER_MIN"] = df["PER"] / df["MIN"].replace(0, 1)

    # Team win %
    win_pcts = {
        "OKC": 0.720, "DEN": 0.610, "MIL": 0.550, "LAL": 0.530, "BOS": 0.720,
        "DAL": 0.560, "PHI": 0.430, "CLE": 0.680, "DET": 0.500, "IND": 0.560,
        "PHX": 0.390, "NYK": 0.610, "MIA": 0.450, "SAC": 0.490, "TOR": 0.290,
        "ORL": 0.530, "ATL": 0.400, "SAS": 0.310, "HOU": 0.530,
    }
    df["W_PCT"] = df["TEAM_ABBREVIATION"].map(win_pcts).fillna(0.500)
    df["WIN_SHARES"] = df["W_PCT"] * (df["PER"] / 15) * (df["MIN"] / 36)
    df["WS48"] = 0.100 + (df["NET_RATING"] / 100)
    return df


# ── Cleaning helper ────────────────────────────────────────────────────────────
def clean_stats(df: pd.DataFrame, min_games: int = 30, min_ppg: float = 23.0) -> pd.DataFrame:
    """
    Filter to legitimate MVP candidates:
      - min_games : eliminates small sample sizes (hardcoded to 65 in app.py)
      - min_ppg   : eliminates non-scorers — 23 PPG cuts role players and
                    second options (e.g. Jalen Williams ~21.6 PPG) who inflate
                    via team-level metrics without being individual MVP caliber
    """
    mask = (df["GP"] >= min_games) & (df["PTS"] >= min_ppg)
    return df[mask].copy().fillna(0)