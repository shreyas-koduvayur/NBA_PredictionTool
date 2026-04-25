"""
mvp_model.py
============
Builds an MVP likelihood score for each player using scikit-learn.

Approach
--------
Real-world MVP voting depends on narrative, team success, and media coverage
— things hard to model from box scores alone. We use a *weighted composite
score* approach:

  1. Select the 7 most MVP-predictive metrics (per basketball research).
  2. Normalise each metric to [0, 1] using MinMaxScaler.
  3. Apply domain-knowledge weights (e.g. PER matters more than STL).
  4. Sum → raw score → convert to probability via softmax.

This gives interpretable, statistically grounded probabilities without
needing historical labelled MVP data.

Sklearn concepts you'll see here:
  - Pipeline          : chain preprocessing + model steps cleanly
  - MinMaxScaler      : scale each column so max=1, min=0
  - fit_transform()   : learn the scale from data, then apply it
  - np.exp / softmax  : convert scores to probabilities that sum to 1
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from data_loader import load_player_stats, clean_stats


# ── MVP metric weights ─────────────────────────────────────────────────────────
# These reflect what basketball analysts historically value most.
# Adjust these values to change how the model ranks candidates.
MVP_WEIGHTS: dict[str, float] = {
    "PER_PER_MIN":  0.25,   # Individual efficiency — captures overall box score impact via PIE
    "WS48":         0.15,   # Win shares per 48 — team-adjusted win contribution
    "PTS":          0.17,   # Scoring volume — rewards scoring titles and high usage
    "AST":          0.13,   # Playmaking — partially in PER, kept as standalone for passers like Jokic
    "TS_PCT":       0.15,   # True shooting % — separates efficient scorers from volume scorers
    "W_PCT":        0.15,   # Team success — voters reward winning but shouldn't inflate role players
}

# Columns we'll pull from the stats DataFrame
METRIC_COLS: list[str] = list(MVP_WEIGHTS.keys())
WEIGHT_ARRAY: np.ndarray = np.array(list(MVP_WEIGHTS.values()))


def _softmax(scores: np.ndarray) -> np.ndarray:
    """
    Convert an array of raw scores into probabilities that sum to 1.

    Amplification factor of 6 concentrates probability on the top 2:
      - 24-25: SGA ~35%, Jokic ~22%  (top2 ~57%)
      - 23-24: Jokic ~26%, Luka ~23% (top2 ~49%)
      - 22-23: Jokic ~35%, Embiid ~26% (top2 ~61%)
    """
    amplified = scores * 6
    exp_scores = np.exp(amplified - amplified.max())
    return exp_scores / exp_scores.sum()


def build_mvp_scores(
    season: str = "2024-25",
    top_n: int = 25,
    min_games: int | None = 30,
) -> tuple[pd.DataFrame, bool]:
    """
    Load stats, score every qualifying player, and return the top_n candidates.

    Parameters
    ----------
    min_games : int or None
        Minimum games played to qualify. If None, uses max(GP) - 20 from the
        data — useful for in-progress seasons where a fixed cutoff doesn't apply.

    Returns
    -------
    (DataFrame, is_mock) — ranked players and whether demo data was used.
    """
    # ── 1. Load & clean raw stats ──────────────────────────────────────────────
    raw_df, is_mock, api_error = load_player_stats(season=season)

    if min_games is None:
        max_gp = int(raw_df["GP"].max())
        min_games = max(max_gp - 20, 1)
        print(f"[mvp_model] Dynamic cutoff: max GP={max_gp} → min_games={min_games}")

    df = clean_stats(raw_df, min_games=min_games)

    # ── 2. Extract the metric matrix ──────────────────────────────────────────
    # Safety: if any metric column is missing (e.g. NET_RATING fetch failed),
    # fill with 0 rather than crashing. The model degrades gracefully.
    for col in METRIC_COLS:
        if col not in df.columns:
            print(f"[mvp_model] Warning: '{col}' missing from data — filling with 0")
            df[col] = 0.0

    X = df[METRIC_COLS].values

    # ── 3. Normalise each metric to [0, 1] ────────────────────────────────────
    # MinMaxScaler transforms each column independently:
    #   scaled_value = (value - col_min) / (col_max - col_min)
    # After scaling, the best player in each metric scores 1.0, worst scores 0.0.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)   # shape: (n_players, 7)

    # Store normalized values back in the DataFrame (useful for radar chart)
    norm_col_names = [f"{col}_NORM" for col in METRIC_COLS]
    # np.ndarray → DataFrame: pd.DataFrame(array, columns=names)
    norm_df = pd.DataFrame(X_scaled, columns=norm_col_names, index=df.index)
    df = pd.concat([df, norm_df], axis=1)  # axis=1 → add as new columns

    # ── 4. Weighted sum → raw MVP score ───────────────────────────────────────
    # np.dot(matrix, vector) → dot product of each row with the weight vector
    # Result shape: (n_players,) — one score per player
    df["MVP_SCORE"] = np.dot(X_scaled, WEIGHT_ARRAY)

    # ── 5. Convert scores to probabilities ────────────────────────────────────
    df["MVP_PROB"] = _softmax(df["MVP_SCORE"].values)

    # ── 6. Sort and rank ──────────────────────────────────────────────────────
    df = df.sort_values("MVP_PROB", ascending=False).reset_index(drop=True)

    # df.index starts at 0 after reset, so rank = index + 1
    df["RANK"] = df.index + 1

    # ── 7. Return only the top_n players ──────────────────────────────────────
    # .head(top_n) returns the first top_n rows (already sorted by probability)
    return df.head(top_n), is_mock, api_error


def get_radar_data(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Extract radar chart data for the top N MVP candidates.

    Returns a dict with:
      - 'players': list of player names
      - 'metrics': list of metric labels (human-readable)
      - 'values':  list of lists — one inner list per player, normalised [0,1]
    """
    top_players = df.head(top_n)

    norm_cols = [f"{col}_NORM" for col in METRIC_COLS]
    metric_labels = ["Efficiency\n(PER)", "Win\nShares", "Points",
                     "Assists", "True\nShooting%", "Team\nWin%"]

    return {
        "players": top_players["PLAYER_NAME"].tolist(),
        "metrics": metric_labels,
        "values": top_players[norm_cols].values.tolist(),
    }