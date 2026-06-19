# 🎮 Steam Game Recommender System 🔍

---

# Live App 🔥

[![Live Demo](https://img.shields.io/badge/Demo-Live%20App-purple?style=for-the-badge&logo=streamlit&logoColor=white)](https://steam-game-recommender-2pfkqz32cvpt2bwlpmkaun.streamlit.app/)

---

A memory-optimized, content-based recommendation engine that suggests video games based on their genres, player tags, categories, and textual descriptions using Steam data. The project features a fast prediction pipeline and a clean, interactive user interface built with Streamlit.

---

## 🛠 Features

- Live Demo: Fully deployed and accessible via Streamlit Cloud. ☁️
- Content-Based Filtering: Uses TfidfVectorizer to extract numerical features from categorical fields and text descriptions. 📝
- Production Optimization: Downsizes a massive 10 GB cosine similarity matrix into a highly compact 7 MB precomputed top-indices integer matrix (int32), allowing seamless deployments on free cloud tiers. 💡
- Robust Search: Supports case-insensitive and partial string matching for user queries. 🔍
- Interactive UI: Built using Streamlit, complete with result size customization and state caching for near-instant responses. ⚡

---

## 📁 Project Structure

steam-game-recommender/
├── app/
│   └── streamlit_app.py     - Streamlit web application interface 🌐
│
├── data/
│   ├── top_indices.npy      - Precomputed Top-50 similar game indices (binary) 💾
│   └── games_cleaned.parquet- Processed metadata storage (lightweight) 💾
│
├── notebooks/
│   ├── 01_eda.ipynb         - Exploratory data analysis 📊
│   └── 02_modeling.ipynb    - Feature extraction, matrix computation, and evaluation ⚙️
│
├── src/
│   ├── preprocessing.py     - Data pipeline routines 🏭
│   └── recommender.py       - Core recommendation matching logic 🤖
│
├── .gitignore               - Excludes raw data and heavy files from version control 🚫
├── pyproject.toml           - Poetry project dependencies configuration 📄
├── poetry.lock              - Deterministic lock file for dependencies 📄
└── README.md                - Project overview and documentation 📄

---

## 🛠 Installation and Setup

This project uses Poetry for dependency and environment management. 📦

1. Clone the repository: ⬇️
git clone [https://github.com/ete9nal/steam-game-recommender.git](https://github.com/ete9nal/steam-game-recommender.git)
cd steam-game-recommender

2. Install dependencies: 🔧
poetry install

3. Activate the environment: 🐍
poetry shell

---

## 🏃 Run the Application Locally

To launch the web interface locally, ensure you are inside the activated Poetry environment and execute:

streamlit run app/streamlit_app.py

The app will compile the layout and automatically launch in your default web browser at http://localhost:8501. 🚀

---

## 🧪 Tech Stack

- Core: Python 3.11 🐍
- Data Processing: Pandas, NumPy, PyArrow (Parquet) 📊
- Machine Learning: Scikit-Learn (TfidfVectorizer, ColumnTransformer, cosine_similarity) ⚙️
- Web UI: Streamlit 🌐
