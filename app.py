"""
app.py
======
NBA MVP Race Predictor — Streamlit application.

Run with:
    streamlit run app.py

Design philosophy: Dark luxury sports analytics.
  - Deep charcoal background, NBA gold (#FDB927) accents
  - Clean data-first layout with high-impact visuals
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from mvp_model import build_mvp_scores, get_radar_data, MVP_WEIGHTS, METRIC_COLS

# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="NBA MVP Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font import ── */
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');

  /* ── Global background & text ── */
  .stApp {
    background-color: #0d0f14;
    color: #e8e8e8;
    font-family: 'Barlow', sans-serif;
  }
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #12151c;
    border-right: 1px solid #1e2330;
  }
  [data-testid="stSidebar"] .stMarkdown p {
    color: #a0a8b8;
    font-size: 0.82rem;
  }

  /* ── Header strip ── */
  .header-strip {
    background: linear-gradient(135deg, #17213a 0%, #0d1526 60%, #17213a 100%);
    border: 1px solid #1e2d4d;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
  }
  .header-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #FDB927;
    letter-spacing: 1px;
    line-height: 1;
    margin: 0;
  }
  .header-sub {
    font-family: 'Barlow', sans-serif;
    font-size: 0.92rem;
    color: #7a8aaa;
    margin: 0;
    margin-top: 0.25rem;
    letter-spacing: 0.3px;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FDB927;
    margin-bottom: 0.6rem;
  }

  /* ── Player rank cards ── */
  .rank-card {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.15s;
  }
  .rank-card:hover { border-color: #FDB927; }
  .rank-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #2a3044;
    min-width: 28px;
    text-align: center;
  }
  .rank-num.gold  { color: #FDB927; }
  .rank-num.silver { color: #9ea8be; }
  .rank-num.bronze { color: #b87333; }
  .rank-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #e8e8e8;
    flex: 1;
  }
  .rank-team {
    font-size: 0.75rem;
    color: #5a6580;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .rank-prob {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: #FDB927;
    min-width: 52px;
    text-align: right;
  }
  .rank-bar-wrap { flex: 0 0 110px; }
  .rank-bar-bg {
    background: #1a1f2e;
    border-radius: 4px;
    height: 6px;
    width: 100%;
  }
  .rank-bar-fill {
    background: linear-gradient(90deg, #c8951e, #FDB927);
    border-radius: 4px;
    height: 100%;
    transition: width 0.3s;
  }

  /* ── Stat pills ── */
  .stat-pill {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #262d3e;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #8090b0;
    margin: 0.15rem;
  }
  .stat-pill span {
    color: #e8e8e8;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin-left: 0.3rem;
  }

  /* ── Metric legend ── */
  .legend-row {
    display: flex;
    justify-content: space-between;
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    color: #7a8aaa;
  }
  .legend-row strong { color: #c0c8d8; }
  .legend-weight { color: #FDB927; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Rank card select buttons ── */
  [data-testid="stHorizontalBlock"] .stButton button {
    background: #1a1f2e;
    border: 1px solid #262d3e;
    color: #5a6580;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    border-radius: 6px;
    padding: 0.3rem 0;
    height: 100%;
    min-height: 62px;
    transition: all 0.15s;
  }
  [data-testid="stHorizontalBlock"] .stButton button:hover {
    border-color: #FDB927;
    color: #FDB927;
    background: #1e2330;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    season = st.selectbox(
        "Season",
        options=["2025-26", "2024-25", "2023-24", "2022-23"],
        index=0,
    )

    top_n_list = st.slider(
        "Players in ranked list",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
    )

    radar_n = st.slider(
        "Players in radar chart",
        min_value=2,
        max_value=8,
        value=5,
        step=1,
    )

    st.markdown("---")
    st.markdown("**Model weights**")
    st.markdown(
        "Adjust these in `mvp_model.py → MVP_WEIGHTS` to change how the model scores players.",
        help="Current weights are based on basketball analytics research.",
    )
    for metric, weight in MVP_WEIGHTS.items():
        st.markdown(
            f'<div class="legend-row"><strong>{metric}</strong>'
            f'<span class="legend-weight">{int(weight * 100)}%</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "Data: [nba_api](https://github.com/swar/nba_api) · "
        "Model: sklearn MinMaxScaler + Softmax"
    )

    st.markdown("---")
    col_r, col_e = st.columns(2)

    with col_r:
        if st.button("🔄 Rerun", use_container_width=True, help="Clear cache and refresh data"):
            st.cache_data.clear()
            st.rerun()

    with col_e:
        if st.button("⏹ Exit", use_container_width=True, help="Shut down the Streamlit server"):
            import os
            os._exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
CURRENT_SEASON = "2025-26"

@st.cache_data(ttl=3600, show_spinner=False)
def get_scores(season):
    """Cache results so re-renders don't re-fetch or re-compute."""
    # Current season: dynamic cutoff (max GP - 20) since it's in progress
    # Past seasons: fixed 60 GP threshold
    min_games = None if season == CURRENT_SEASON else 60
    return build_mvp_scores(
        season=season,
        top_n=25,
        min_games=min_games,
    )


with st.spinner("Loading NBA stats…"):
    df, is_mock, api_error = get_scores(season)

radar_data = get_radar_data(df, top_n=radar_n)
top_players = df.head(top_n_list)
max_prob = df["MVP_PROB"].max()

# ── Selected player state — defaults to the #1 ranked player ──────────────────
if "selected_player" not in st.session_state:
    st.session_state.selected_player = df.iloc[0]["PLAYER_NAME"]

# Resolve the selected row (fall back to #1 if name no longer in filtered list)
selected_rows = df[df["PLAYER_NAME"] == st.session_state.selected_player]
selected = selected_rows.iloc[0] if not selected_rows.empty else df.iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
data_source_badge = (
    '<span style="background:#2a1a00;color:#f0a500;border:1px solid #7a4a00;'
    'border-radius:4px;padding:1px 8px;font-size:0.78rem;font-weight:700;'
    'letter-spacing:0.5px;margin-left:6px">⚠ DEMO DATA</span>'
    if is_mock else
    '<span style="background:#0a2a0a;color:#4caf50;border:1px solid #1a5c1a;'
    'border-radius:4px;padding:1px 8px;font-size:0.78rem;font-weight:700;'
    'letter-spacing:0.5px;margin-left:6px">● LIVE</span>'
)

st.markdown(f"""
<div class="header-strip">
  <div style="font-size:3rem;line-height:1">🏀</div>
  <div>
    <p class="header-title">MVP RACE PREDICTOR</p>
    <p class="header-sub">
      {season} Regular Season &nbsp;·&nbsp;
      {len(df)} players ({int(df['GP'].min())}+ games) &nbsp;·&nbsp;
      stats.nba.com {data_source_badge}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

if is_mock and api_error:
    with st.expander("⚠️ Could not reach stats.nba.com — showing demo data. Click to see why."):
        st.code(api_error, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# Layout: Radar + Ranked List side by side
# ══════════════════════════════════════════════════════════════════════════════
col_radar, col_list = st.columns([1.15, 1], gap="large")


# ── LEFT: Radar / Spider Chart ─────────────────────────────────────────────────
with col_radar:
    st.markdown('<p class="section-label">📡 MVP Candidate Comparison</p>', unsafe_allow_html=True)

    # Colour palette — one per player
    COLORS = ["#FDB927", "#4fc3f7", "#ef5350", "#66bb6a", "#ab47bc", "#ff7043", "#26c6da", "#d4e157"]

    # Plotly radar chart
    fig = go.Figure()

    metrics_display = ["PER", "Win Shares", "Points", "Assists", "True Shooting%", "Team Win%"]

    for i, player in enumerate(radar_data["players"]):
        vals = radar_data["values"][i]
        # Radar charts need the first value repeated at the end to close the polygon
        vals_closed = vals + [vals[0]]
        labels_closed = metrics_display + [metrics_display[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({int(COLORS[i][1:3], 16)}, {int(COLORS[i][3:5], 16)}, {int(COLORS[i][5:], 16)}, 0.12)",
            line=dict(color=COLORS[i], width=2.5),
            name=player.split()[-1],   # last name only keeps legend compact
            hovertemplate=(
                "<b>%{theta}</b><br>"
                f"{player}<br>"
                "Normalised: %{r:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#0d0f14",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="#1e2330",
                linecolor="#1e2330",
            ),
            angularaxis=dict(
                tickfont=dict(family="Barlow Condensed", size=13, color="#9ea8be"),
                gridcolor="#1e2330",
                linecolor="#1e2330",
            ),
        ),
        paper_bgcolor="#12151c",
        plot_bgcolor="#12151c",
        legend=dict(
            font=dict(family="Barlow Condensed", size=13, color="#c0c8d8"),
            bgcolor="#0d0f14",
            bordercolor="#1e2330",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=60, l=60, r=60),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Stat breakdown — updates when a player is clicked in the ranked list ────
    st.markdown(f'<p class="section-label">📊 {selected["PLAYER_NAME"].upper()} — STAT BREAKDOWN</p>', unsafe_allow_html=True)

    stat_html = ""
    stat_map = {
        "PTS": "PPG", "AST": "APG", "REB": "RPG",
        "STL": "SPG", "BLK": "BPG", "PER": "PER",
        "TS_PCT": "TS%", "WS48": "WS/48",
        "W_PCT": "Team Win%", "GP": "Games",
    }
    for col, label in stat_map.items():
        val = selected[col]
        if col == "GP":
            fmt = f"{int(val)}"
        elif col in ("W_PCT", "TS_PCT"):
            fmt = f"{val:.1%}"
        else:
            fmt = f"{val:.1f}"
        stat_html += f'<div class="stat-pill">{label}<span>{fmt}</span></div>'

    st.markdown(stat_html, unsafe_allow_html=True)


# ── RIGHT: Ranked List ─────────────────────────────────────────────────────────
with col_list:
    st.markdown('<p class="section-label">🏆 MVP Probability Rankings</p>', unsafe_allow_html=True)

    rank_colors = {1: "gold", 2: "silver", 3: "bronze"}

    for _, row in top_players.iterrows():
        rank     = int(row["RANK"])
        name     = row["PLAYER_NAME"]
        team     = row["TEAM_ABBREVIATION"]
        prob_pct = row["MVP_PROB"] * 100
        bar_w    = (row["MVP_PROB"] / max_prob) * 100
        r_class  = rank_colors.get(rank, "")
        is_selected = (name == st.session_state.selected_player)
        border_color = "#FDB927" if is_selected else "#1e2330"

        # Each card is two columns: the info block (HTML) + a slim select button
        c_card, c_btn = st.columns([11, 2])

        with c_card:
            st.markdown(f"""
<div class="rank-card" style="border-color:{border_color};margin-bottom:0">
  <div class="rank-num {r_class}">{rank}</div>
  <div style="flex:1;min-width:0">
    <div class="rank-name">{name}</div>
    <div class="rank-team">{team} &nbsp;·&nbsp;
      {row['PTS']:.1f} PPG &nbsp;·&nbsp;
      {row['REB']:.1f} RPG &nbsp;·&nbsp;
      {row['AST']:.1f} APG
    </div>
    <div style="margin-top:0.45rem">
      <div class="rank-bar-bg">
        <div class="rank-bar-fill" style="width:{bar_w:.1f}%"></div>
      </div>
    </div>
  </div>
  <div class="rank-prob">{prob_pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        with c_btn:
            btn_label = "✓" if is_selected else "View"
            if st.button(btn_label, key=f"sel_{rank}", use_container_width=True):
                st.session_state.selected_player = name
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Bottom: Full data table (collapsible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📋 Full Stats Table (all qualifying players)"):
    display_cols = ["RANK", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP",
                    "PTS", "AST", "REB", "STL", "BLK",
                    "PER", "WS48", "MVP_SCORE", "MVP_PROB"]

    display_df = df[display_cols].copy()
    display_df["MVP_PROB"] = (display_df["MVP_PROB"] * 100).round(1).astype(str) + "%"
    display_df["MVP_SCORE"] = display_df["MVP_SCORE"].round(3)
    display_df = display_df.rename(columns={
        "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team",
        "MVP_SCORE": "Score", "MVP_PROB": "Probability",
        "WS48": "Win Shares",
    })

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "RANK": st.column_config.NumberColumn("#", width="small"),
            "Score": st.column_config.NumberColumn(format="%.3f"),
            "PTS": st.column_config.NumberColumn("PPG", format="%.1f"),
            "AST": st.column_config.NumberColumn("APG", format="%.1f"),
            "REB": st.column_config.NumberColumn("RPG", format="%.1f"),
            "STL": st.column_config.NumberColumn("SPG", format="%.1f"),
            "BLK": st.column_config.NumberColumn("BPG", format="%.1f"),
            "PER": st.column_config.NumberColumn("PER", format="%.1f"),
            "Win Shares": st.column_config.NumberColumn("WS/48", format="%.3f"),
        },
    )