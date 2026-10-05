# NLP Movie Recommendation System

Content-based movie recommender using **TF-IDF + Cosine Similarity**, with a weighted
multi-feature engine and a Streamlit UI.

## Files
- `movie_recommender.ipynb` — **Level 1**: single merged TF-IDF over genre+keywords+overview. Avg Precision@5 = 0.71
- `movie_recommender_level2.ipynb` — **Level 2**: separate TF-IDF per feature (plot/genre/keyword/cast/director), combined with weights (50/20/15/10/5). Avg Precision@5 = 0.77 (50/20/15/10/5 weights). Also includes a weight-tuning experiment comparing 3 weight configurations.
- `app.py` — **Level 3**: Streamlit UI with an engine toggle: **Weighted (Level 2, default)** or **Basic (Level 1)**, adjustable feature weights, per-result score breakdown
- `data/tmdb_5000_movies.csv`, `data/tmdb_5000_credits.csv` — TMDB 5000 dataset (movies + cast/crew)
- `movies.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` — Level 1 saved artifacts (used by the app's Basic mode)
- `movies_v2.pkl` — Level 2 saved artifacts (dict containing the dataframe + 5 vectorizers + 5 matrices + weights)

## How to run the notebooks
1. Keep `data/tmdb_5000_movies.csv` and `data/tmdb_5000_credits.csv` in the `data/` subfolder.
2. `pip install pandas numpy scikit-learn nltk jupyter ipykernel`
3. Run `movie_recommender.ipynb` first (Level 1), then `movie_recommender_level2.ipynb` (Level 2) — Run All on each.

## How to run the web app
1. Make sure all four artifact sets exist next to `app.py`: `movies.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` (Level 1) and `movies_v2.pkl` (Level 2).
2. `pip install streamlit scikit-learn nltk pandas numpy pyarrow` (pyarrow is needed to unpickle `movies_v2.pkl` if it was saved with pandas 3)
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

## About the match percentage in the app
The gauge shows **relative relevance**: each result's score divided by the best score for that query
across all movies (so the top pick for a query is ~100%). Raw cosine similarity between a 4-word query and a
long movie document is always small (often 20-40%) and is only meaningful for *ranking*, so it is not displayed directly.
In Weighted mode, if the query names no actor/director, the cast/director weights are dropped and the
remaining weights renormalised (otherwise every score would be capped at 85%).

## Known limitations (good to state in the report)
- Evaluation uses 7 genre-keyword queries scored by "does the result contain the target genre", which is
  partly circular (genre is also a search field). Compound queries like "romantic comedy" are scored on one
  genre only. A manually-judged query set would be a stronger evaluation.
- Director weight is 5%, so naming a director only nudges the ranking.
- Lemmatisation does not link "romantic" to "romance"; embeddings would.

## Project levels
- **Level 1 (done)** — `movie_recommender.ipynb`
- **Level 2 (done)** — `movie_recommender_level2.ipynb`
- **Level 3 (done)** — `app.py` (Streamlit UI, Basic vs Weighted engines)

## Optional next step
Add sentence-embedding similarity (e.g. `all-MiniLM-L6-v2`) as a third engine and compare all three on an
expanded, manually-judged evaluation set.
