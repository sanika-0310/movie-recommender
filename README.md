# NLP Movie Recommendation System

A content-based movie recommender built with **TF-IDF + Cosine Similarity**, featuring a
weighted multi-feature scoring engine and a Streamlit web UI.

## Overview
This project recommends movies based on content similarity — not user ratings or collaborative
filtering — using the TMDB 5000 dataset. It was built in two iterations to explore how feature
engineering affects recommendation quality, going from a single merged text blob to a weighted
multi-feature scoring model.

## Approach

**V1 — Single TF-IDF blob**
Genre, keywords, and overview are merged into one text field per movie, then scored with a
single TF-IDF + cosine similarity model.
→ Avg Precision@5: **0.71**

**V2 — Weighted multi-feature model**
Five fields (plot, genre, keywords, cast, director) are kept separate, each scored with its own
TF-IDF + cosine similarity, then blended using explicit weights:

| Feature   | Weight |
|-----------|--------|
| Plot      | 50%    |
| Genre     | 20%    |
| Keywords  | 15%    |
| Cast      | 10%    |
| Director  | 5%     |

This approach is more explainable (each factor's contribution to a match can be reported) and
tunable — a small experiment in the notebook compares this weighting against "plot-heavy" and
"genre-heavy" alternatives, measured by Precision@5.
→ Avg Precision@5: **0.74**

**Debugging note:** an early version of V2 collapsed multi-word genre/keyword phrases into single
tokens (e.g. `"Science Fiction"` → `"sciencefiction"`), which silently broke matching for any
naturally-typed query. Fixed by preserving spacing and using `ngram_range=(1,2)` so phrase matches
like "science fiction" work correctly. Precision@5 on the sci-fi test query went from **0.40 → 1.00**
after the fix.

## Files
- `movie_recommender.ipynb` — V1: single merged TF-IDF model
- `movie_recommender_level2.ipynb` — V2: weighted multi-feature model + weight-tuning experiment
- `app.py` — Streamlit UI (currently using V1 artifacts)
- `data/tmdb_5000_movies.csv`, `data/tmdb_5000_credits.csv` — TMDB 5000 dataset
- `movies.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` — V1 saved artifacts
- `movies_v2.pkl` — V2 saved artifacts (dataframe + 5 vectorizers + 5 matrices + weights)

## Setup & Usage

### Notebooks
```bash
pip install pandas numpy scikit-learn nltk jupyter ipykernel
```
Keep the two CSVs in `data/`, then run `movie_recommender.ipynb` first, followed by
`movie_recommender_level2.ipynb` (Run All on each).

### Web app
```bash
pip install streamlit scikit-learn nltk pandas
streamlit run app.py
```
