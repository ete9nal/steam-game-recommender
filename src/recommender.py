"""
Content-based game recommendation logic.

Recommendations are based on a precomputed top-50 similarity index
(see notebooks/recsys.ipynb for how it's built from TF-IDF + cosine similarity).
"""

import numpy as np
import pandas as pd


def get_recommendations(
    game_title: str,
    df: pd.DataFrame,
    top_indices_matrix: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the top_n games most similar to game_title.

    Parameters
    ----------
    game_title : partial or full game name (case-insensitive)
    df : dataframe with a 'name' column, index must match the rows
         used to build top_indices_matrix (i.e. reset_index(drop=True))
    top_indices_matrix : precomputed (n_games, top_k) array of similar-game
         row positions, sorted by similarity descending. Column 0 is the
         game itself.
    top_n : how many recommendations to return

    Returns
    -------
    DataFrame with the recommended games' rows (same columns as df).
    Empty DataFrame if no match is found.
    """
    matches = df[df['name'].str.contains(game_title, case=False, na=False)]

    if matches.empty:
        return pd.DataFrame(columns=df.columns)

    # Positional index into df — df must be reset_index(drop=True) for this
    # to line up with top_indices_matrix rows.
    position = df.index.get_loc(matches.index[0])

    similar_positions = top_indices_matrix[position, 1:top_n + 1]
    return df.iloc[similar_positions].reset_index(drop=True)


def get_popularity_baseline(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the top_n most positively-reviewed games, ignoring content similarity."""
    return df.nlargest(top_n, 'positive').reset_index(drop=True)
