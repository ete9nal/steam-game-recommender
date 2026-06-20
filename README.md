# 🎮 Steam Game Recommender

A content-based recommendation system that helps users discover new Steam games
based on the similarity of game descriptions, genres, and tags.

**Live demo:** [https://steam-game-recommender-2pfkqz32cvpt2bwlpmkaun.streamlit.app/](https://steam-game-recommender-rd6atpmbwwhcdmuzngynbq.streamlit.app/)

## Problem

With 90,000+ games on Steam, finding a new game similar to one you already love is
hard. This project builds a recommendation engine that, given a game you enjoy,
suggests similar titles based on what the game *is* (genre, tags, description) —
no user rating history required.

## Approach

**Content-based filtering** using TF-IDF + cosine similarity.

The dataset contains only game metadata (no user-item interaction data — no
ratings, purchase history, or playtime per user), which rules out collaborative
filtering. Content-based filtering is the right fit for the data available, and
has the added benefit of working immediately for newly released games (no
cold-start problem).

### Pipeline

1. **Data loading & overview** — load and inspect the Steam Games Dataset 2025
2. **EDA** — explore price, release date, and review distributions; choose a
   review-count threshold to filter out low-signal entries
3. **Preprocessing & feature engineering** — combine `genres`, `tags`,
   `categories`, and `about_the_game` into a single cleaned text field
4. **Modeling** — vectorize text with `TfidfVectorizer`, compute pairwise
   cosine similarity, keep the top-50 most similar games per title
5. **Evaluation** — qualitative check against known titles, comparison with a
   popularity baseline, catalog coverage analysis
6. **Streamlit app** — interactive UI to search a game and view recommendations

## Dataset

[Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
(94,948 games), filtered to 36,259 games with at least 30 reviews.

## Tech Stack

- `pandas`, `numpy` — data handling
- `scikit-learn` — `TfidfVectorizer`, `cosine_similarity`
- `streamlit` — web app
- `matplotlib`, `seaborn` — EDA visualizations

## Project Structure

```
steam-game-recommender/
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── data/
│   ├── games_march2025_full.csv   # raw dataset (not tracked in git)
│   ├── games_cleaned.parquet      # processed games for the app
│   └── top_indices.npy            # precomputed top-50 similarity indices
├── notebooks/
│   └── recsys.ipynb           # full EDA → modeling → evaluation pipeline
├── src/
│   └── recommender.py         # recommendation logic
├── pyproject.toml
└── README.md
```

## Running locally

```bash
poetry install
poetry run jupyter notebook   # run notebooks/recsys.ipynb to generate data artifacts
poetry run streamlit run app/streamlit_app.py
```

## Why not collaborative filtering?

The dataset has no per-user interaction data (ratings, purchases, playtime by
user), which collaborative filtering and matrix factorization methods (SVD,
ALS) require. Building a synthetic user-item matrix (e.g. treating genres as
"users") would not reflect real user behavior and was deliberately avoided in
favor of an honest, data-appropriate content-based approach.

## Future improvements

- Replace TF-IDF with sentence embeddings (`sentence-transformers`) for
  better semantic similarity
- Add a hybrid score that blends content similarity with review quality
- If user interaction data becomes available, add collaborative filtering
  (ALS via `implicit`) as a complementary signal
