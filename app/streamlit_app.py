import os
import numpy as np
import pandas as pd
import streamlit as st
from src.recommender import get_recommendations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="centered"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
PARQUET_PATH = os.path.join(DATA_DIR, 'games_cleaned.parquet')
MATRIX_PATH  = os.path.join(DATA_DIR, 'top_indices.npy')

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df          = pd.read_parquet(PARQUET_PATH)
    top_indices = np.load(MATRIX_PATH)
    return df, top_indices

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎮 Steam Game Recommender")
st.write(
    "Find your next favorite game based on genres, tags, and descriptions "
    "using Content-Based Filtering."
)

# Check artifacts exist
if not os.path.exists(PARQUET_PATH) or not os.path.exists(MATRIX_PATH):
    st.error(
        "Missing data artifacts. "
        "Run the notebook first to generate `games_cleaned.parquet` and `top_indices.npy`."
    )
    st.stop()

df, top_indices = load_data()

st.markdown("---")

# Game selector
selected_game = st.selectbox(
    "Select or type a game you enjoy:",
    options=df['name'].unique(),
    index=0
)

# Number of recommendations slider
top_n = st.slider("Number of recommendations:", min_value=3, max_value=20, value=5)

st.markdown("---")

# Generate button
if st.button("🔍 Find Similar Games", type="primary"):
    with st.spinner("Finding similar games..."):
        recommendations = get_recommendations(selected_game, df, top_indices, top_n=top_n)

    if recommendations.empty:
        st.warning(f'No recommendations found for **{selected_game}**.')
    else:
        st.subheader(f"Top {top_n} games similar to **{selected_game}**:")

        for i, game in enumerate(recommendations, 1):
            # Look up game details
            game_row = df[df['name'] == game]

            if not game_row.empty:
                row = game_row.iloc[0]
                col1, col2 = st.columns([1, 4])

                with col1:
                    if pd.notna(row.get('header_image')):
                        st.image(row['header_image'], use_container_width=True)

                with col2:
                    st.markdown(f"**{i}. {game}**")

                    # Price
                    price = row.get('price', None)
                    if pd.notna(price):
                        price_str = "Free" if price == 0 else f"${price:.2f}"
                        st.caption(f"💰 {price_str}")

                    # Review score
                    pct = row.get('pct_pos_total', -1)
                    if pct > 0:
                        if pct >= 80:
                            emoji = "🟢"
                        elif pct >= 60:
                            emoji = "🟡"
                        else:
                            emoji = "🔴"
                        st.caption(f"{emoji} {pct}% positive reviews")

            else:
                st.markdown(f"**{i}. {game}**")

            st.markdown("---")