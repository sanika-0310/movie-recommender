# NLP Movie Recommendation System

Content-based movie recommender using **TF-IDF + Cosine Similarity**, with a weighted
multi-feature engine and a Streamlit UI.

## Files
- `movie_recommender.ipynb` — **Level 1**: single merged TF-IDF over genre+keywords+overview. Avg Precision@5 = 0.71
- `movie_recommender_level2.ipynb` — **Level 2**: separate TF-IDF per feature (plot/genre/keyword/cast/director), combined with weights (50/20/15/10/5). Avg Precision@5 = 0.74. Also includes a weight-tuning experiment comparing 3 weight configurations.
- `app.py` — **Level 3**: Streamlit UI (currently wired to the Level 1 artifacts — see note below to switch it to Level 2)
- `data/tmdb_5000_movies.csv`, `data/tmdb_5000_credits.csv` — TMDB 5000 dataset (movies + cast/crew)
- `movies.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` — Level 1 saved artifacts
- `movies_v2.pkl` — Level 2 saved artifacts (dict containing the dataframe + 5 vectorizers + 5 matrices + weights)

## How to run the notebooks
1. Keep `data/tmdb_5000_movies.csv` and `data/tmdb_5000_credits.csv` in the `data/` subfolder.
2. `pip install pandas numpy scikit-learn nltk jupyter ipykernel`
3. Run `movie_recommender.ipynb` first (Level 1), then `movie_recommender_level2.ipynb` (Level 2) — Run All on each.

## How to run the web app
1. Make sure `movies.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` exist (from running Level 1).
2. `pip install streamlit scikit-learn nltk pandas`
3. `streamlit run app.py`
4. Opens at `http://localhost:8501`.

## Level 1 vs Level 2 — what changed
Level 1 merges genre + keywords + overview into **one** text blob per movie and runs a single TF-IDF.
Level 2 keeps **five separate fields** (plot, genre, keywords, cast, director), scores each with its
own TF-IDF + cosine similarity, then blends the five scores with explicit weights:

| Feature   | Weight |
|-----------|--------|
| Plot      | 50%    |
| Genre     | 20%    |
| Keywords  | 15%    |
| Cast      | 10%    |
| Director  | 5%     |

This is more explainable (you can report *how much* each factor contributed to a match) and tunable —
the notebook includes a small experiment comparing this weighting against a "plot-heavy" and a
"genre-heavy" alternative, measured by Precision@5.

**Important bug fix along the way:** the first version of Level 2 squashed multi-word genre/keyword
names into single tokens (`"Science Fiction"` → `"sciencefiction"`), which silently broke matching for
any query written the normal way. Fixed by keeping natural spacing and using `ngram_range=(1,2)` so
phrase matches like "science fiction" still work as a unit. Precision@5 went from a broken 0.40 to 1.00
on the sci-fi test query after the fix — worth mentioning in your report as a debugging example.

## Project levels
- **Level 1 (done)** — `movie_recommender.ipynb`
- **Level 2 (done)** — `movie_recommender_level2.ipynb`
- **Level 3 (done)** — `app.py` (Streamlit UI, currently using Level 1 artifacts)

## Optional next step
Wire `app.py` to `movies_v2.pkl` and add a toggle ("Basic" vs "Weighted") so the demo can show both
engines side by side — good for the viva if you're asked to justify the weighting choice.
