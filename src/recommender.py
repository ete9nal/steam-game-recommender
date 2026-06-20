import numpy as np
import pandas as pd


def get_recommendations(game_title: str, df: pd.DataFrame, top_indices_matrix: np.ndarray, top_n: int = 5) -> pd.Series:
    """
    Fast and memory-efficient recommendation using a precomputed top-indices matrix.
    Handles case-insensitive and partial string matching.
    """
    # 1. Find the game by partial name match (case-insensitive)
    matches = df[df['name'].str.contains(game_title, case=False, na=False)]

    if matches.empty:
        print(f"Game '{game_title}' not found in the dataset.")
        return pd.Series(dtype='object')

    # Get the internal index of the first matching game
    idx = matches.index[0]
    pd_idx = df.index.get_loc(idx)

    # 2. Get precomputed top recommendations for this game index
    # Exclude the game itself from recommendations and take top_n results
    game_top_indices = top_indices_matrix[pd_idx]
    game_top_indices = game_top_indices[game_top_indices != pd_idx][:top_n]

    # 3. Return the game titles using the fast .iloc indexer
    return df['name'].iloc[game_top_indices]