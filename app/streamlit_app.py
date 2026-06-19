import os
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="centered"
)

# Define paths to data artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PARQUET_PATH = os.path.join(DATA_DIR, 'games_cleaned.parquet')
MATRIX_PATH = os.path.join(DATA_DIR, 'top_indices.npy')


@st.cache_data
def load_data():
    """
    Loads the precomputed recommendation artifacts.
    Cached to prevent reloading on every user interaction.
    """
    df = pd.read_parquet(PARQUET_PATH)
    top_indices = np.load(MATRIX_PATH)
    return df, top_indices


def get_recommendations(game_title: str, df: pd.DataFrame, top_indices_matrix: np.ndarray, top_n: int = 5) -> pd.Series:
    """
    Finds recommendations based on partial matching and precomputed top indices.
    """
    # Case-insensitive partial string matching
    matches = df[df['name'].str.contains(game_title, case=False, na=False)]

    if matches.empty:
        return pd.Series(dtype='object')

    idx = matches.index[0]

    # Extract precomputed top items (excluding the selected game itself if possible)
    # Slicing from 1 to top_n + 1 assumes index 0 is the query game itself
    game_top_indices = top_indices_matrix[idx, 1:top_n + 1]

    return df['name'].iloc[game_top_indices]


# --- UI Layout ---

st.title("🎮 Steam Game Recommender System")
st.write("Find your next favorite game based on genres, tags, and descriptions using Content-Based Filtering.")

# Load the data artifacts safely
if not os.path.exists(PARQUET_PATH) or not os.path.exists(MATRIX_PATH):
    st.error(
        "Missing data artifacts! Please ensure 'games_cleaned.parquet' and 'top_indices.npy' exist inside the 'data/' folder.")
else:
    df, top_indices = load_data()

    # Dropdown with search capability containing all unique game names
    all_games = df['name'].unique()
    selected_game = st.selectbox(
        "Select or type a game you love:",
        options=all_games,
        index=0
    )

    # Slider to customize the number of recommendations
    top_n = st.slider("Number of recommendations:", min_value=3, max_value=20, value=5)

    st.markdown("---")

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Finding similar games..."):
            recommendations = get_recommendations(selected_game, df, top_indices, top_n=top_n)

            if not recommendations.empty:
                st.subheader(f"Top {top_n} games similar to **{selected_game}**:")
                for i, game in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {game}")
            else:
                st.warning("Could not generate recommendations for the selected game.")