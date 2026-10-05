"""
CineMatch — NLP Movie Recommendation System (Streamlit UI, Level 3)

Two engines, switchable in the UI:
  * Basic (Level 1)    — one merged TF-IDF over genre + keywords + plot
                         (movies.pkl, tfidf_vectorizer.pkl, tfidf_matrix.pkl)
  * Weighted (Level 2) — separate TF-IDF per feature (plot / genre / keywords),
                         plus cast & director name matching, blended with
                         adjustable weights (movies_v2.pkl)

Match % is shown RELATIVE to the best possible match for the query (the top
hit for a query is ~100%). Raw cosine similarity between a short query and a
long movie document is always small, so it is only meaningful for ranking.
"""

import re
import pickle

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CineMatch", page_icon="🎞️", layout="wide")

CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0E0C10;
    --bg-card: #1B1720;
    --bg-card-hover: #221D28;
    --gold: #E3B23C;
    --gold-soft: rgba(227, 178, 60, 0.14);
    --gold-border: rgba(227, 178, 60, 0.35);
    --text-primary: #F3EDE4;
    --text-muted: #9C9289;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 50% -10%, rgba(227,178,60,0.10), transparent 60%),
        var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* ---------- Hero ---------- */
.cinema-hero {
    text-align: center;
    padding: 1.6rem 0 0.4rem 0;
}
.cinema-hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.2rem;
    letter-spacing: 0.06em;
    color: var(--gold);
    text-shadow: 0 0 28px rgba(227,178,60,0.35);
    margin: 0;
    line-height: 1;
}
.cinema-hero p {
    color: var(--text-muted);
    font-size: 1.02rem;
    margin-top: 0.4rem;
}

/* ---------- Inputs ---------- */
.stTextInput input {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--gold-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 1.05rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus {
    box-shadow: 0 0 0 2px var(--gold-soft) !important;
    border-color: var(--gold) !important;
}
.stTextInput input::placeholder { color: var(--text-muted) !important; }

.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--gold-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stSlider [data-baseweb="slider"] > div > div { background: var(--gold) !important; }
.stSlider [role="slider"] {
    background-color: var(--gold) !important;
    border-color: var(--gold) !important;
}

/* ---------- Button ---------- */
.stButton > button {
    background: linear-gradient(135deg, #E3B23C, #C6902A) !important;
    color: #17130D !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    padding: 0.6rem 1.6rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(227,178,60,0.30);
}

/* ---------- Expander (filters) ---------- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--gold-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] {
    border: 1px solid var(--gold-border) !important;
    border-radius: 8px !important;
    background-color: transparent !important;
}

/* ---------- Section heading ---------- */
.now-showing {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    letter-spacing: 0.05em;
    color: var(--text-primary);
    border-bottom: 1px solid var(--gold-border);
    padding-bottom: 0.35rem;
    margin: 1.6rem 0 1rem 0;
}

/* ---------- Ticket card ---------- */
.ticket-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    gap: 1.5rem;
    align-items: center;
}
.ticket-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.7rem;
    letter-spacing: 0.02em;
    color: var(--text-primary);
    margin: 0;
}
.ticket-meta {
    margin: 0.25rem 0 0.7rem 0;
}
.ticket-pill {
    display: inline-block;
    background: var(--gold-soft);
    border: 1px solid var(--gold-border);
    color: var(--gold);
    border-radius: 100px;
    padding: 0.12rem 0.65rem;
    font-size: 0.78rem;
    margin-right: 0.35rem;
    margin-bottom: 0.2rem;
}
.ticket-year-rating {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-right: 0.6rem;
}
.ticket-overview {
    color: var(--text-primary);
    opacity: 0.88;
    font-size: 0.94rem;
    line-height: 1.5;
    margin: 0.3rem 0 0.6rem 0;
}
.because-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-right: 0.4rem;
}
.reason-chip {
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.14);
    color: var(--text-muted);
    border-radius: 6px;
    padding: 0.08rem 0.5rem;
    font-size: 0.78rem;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
}

/* ---------- Match gauge ---------- */
.match-gauge { text-align: center; flex-shrink: 0; }
.match-gauge svg { display: block; margin: 0 auto; }
.match-gauge .pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.15rem;
    fill: var(--gold);
}
.match-gauge .label {
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    margin-top: 0.15rem;
}
/* ---------- Engine toggle / breakdown ---------- */
.stRadio label, .stRadio p { color: var(--text-primary) !important; }
.breakdown-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-right: 0.4rem;
}
.score-chip {
    display: inline-block;
    background: var(--gold-soft);
    border: 1px solid var(--gold-border);
    color: var(--gold);
    border-radius: 6px;
    padding: 0.08rem 0.5rem;
    font-size: 0.76rem;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
}
</style>
"""


st.markdown(CINEMA_CSS, unsafe_allow_html=True)

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


BASIC = "Basic (Level 1 · merged TF-IDF)"
WEIGHTED = "Weighted (Level 2 · multi-feature)"
FEATURES = ["plot", "genre", "keyword", "cast", "director"]
FEATURE_LABELS = {"plot": "Plot", "genre": "Genre", "keyword": "Keywords",
                  "cast": "Cast", "director": "Director"}
DEFAULT_WEIGHTS = {"plot": 0.50, "genre": 0.20, "keyword": 0.15, "cast": 0.10, "director": 0.05}


@st.cache_resource
def load_artifacts():
    # Level 1
    with open("movies.pkl", "rb") as f:
        movies_v1 = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_v1 = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        matrix_v1 = pickle.load(f)
    # Level 2 (dict with dataframe + 5 vectorizers + 5 matrices + weights)
    with open("movies_v2.pkl", "rb") as f:
        v2 = pickle.load(f)
    return movies_v1, tfidf_v1, matrix_v1, v2


movies_v1, tfidf_v1, matrix_v1, V2 = load_artifacts()

# The Level 2 dataframe is the master list shown in the UI (it has cast + director).
movies = V2["movies"].reset_index(drop=True)
all_genres = sorted({g for genre_list in movies["genres_list"] for g in genre_list})
saved_weights = V2.get("weights") or DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def basic_scores(query: str) -> np.ndarray:
    """Level 1: cosine similarity against the merged TF-IDF matrix, aligned to `movies` by id."""
    q = tfidf_v1.transform([clean_text(query)])
    sims = cosine_similarity(q, matrix_v1).flatten()
    aligned = pd.Series(sims, index=movies_v1["id"].values)
    aligned = aligned[~aligned.index.duplicated()]
    return aligned.reindex(movies["id"].values).fillna(0.0).to_numpy()


def name_hits(query: str) -> tuple[np.ndarray, np.ndarray]:
    """1.0 where an actor / the director is named in the query (names compared with spaces removed)."""
    q = re.sub(r"[^a-z]", "", query.lower())
    cast_hit = movies["cast_text"].apply(
        lambda s: 1.0 if s and any(len(n) >= 5 and n in q for n in s.split()) else 0.0
    ).to_numpy()
    dir_hit = movies["director_text"].apply(
        lambda d: 1.0 if d and len(d) >= 5 and d in q else 0.0
    ).to_numpy()
    return cast_hit, dir_hit


def weighted_scores(query: str, weights: dict) -> tuple[np.ndarray, dict, dict]:
    """
    Level 2: blend per-feature similarities.
    Returns (final_score, per-feature weighted contributions, weights actually used).

    If the query names no actor or director, the cast/director weights would just
    be dead weight (capping every score at 85%), so they are dropped and the
    remaining weights are renormalised.
    """
    qc = clean_text(query)
    sims = {
        "plot": cosine_similarity(V2["tfidf_plot"].transform([qc]), V2["matrix_plot"]).flatten(),
        "genre": cosine_similarity(V2["tfidf_genre"].transform([qc]), V2["matrix_genre"]).flatten(),
        "keyword": cosine_similarity(V2["tfidf_keyword"].transform([qc]), V2["matrix_keyword"]).flatten(),
    }
    sims["cast"], sims["director"] = name_hits(query)

    used = dict(weights)
    if sims["cast"].sum() == 0:
        used["cast"] = 0.0
    if sims["director"].sum() == 0:
        used["director"] = 0.0
    total = sum(used.values())
    used = {k: v / total for k, v in used.items()} if total > 0 else dict(DEFAULT_WEIGHTS)

    contrib = {f: used[f] * sims[f] for f in FEATURES}
    final = sum(contrib.values())
    return final, contrib, used


def recommend(query: str, mode: str = WEIGHTED, top_n: int = 5, genre_filter: str | None = None,
              min_rating: float = 0.0, weights: dict | None = None) -> pd.DataFrame:
    contrib = None
    if mode == BASIC:
        scores = basic_scores(query)
    else:
        scores, contrib, _ = weighted_scores(query, weights or saved_weights)

    # Relative match: best possible score for this query across ALL movies (before filters),
    # so filters can't inflate the percentages.
    best = float(scores.max()) if len(scores) else 0.0
    relevance = scores / best * 100 if best > 0 else np.zeros_like(scores)

    result = movies.copy()
    result["similarity"] = scores
    result["similarity_%"] = np.round(relevance, 1)
    if contrib is not None:
        for f in FEATURES:
            result[f"c_{f}"] = contrib[f]

    if genre_filter and genre_filter != "All":
        result = result[result["genres_list"].apply(lambda g: genre_filter in g)]
    if min_rating > 0:
        result = result[result["vote_average"] >= min_rating]

    result = result[result["similarity"] > 0]  # don't pad with zero-score movies
    result = result.sort_values("similarity", ascending=False).head(top_n)
    return result.reset_index(drop=True)


def why_recommended(query: str, row: pd.Series) -> list[str]:
    """Matched genres / keywords / names / plot words shared between the query and this movie."""
    query_tokens = set(clean_text(query).split())
    q_lower = query.lower()
    reasons = []

    reasons.extend(g for g in row["genres_list"] if g.lower() in q_lower)
    reasons.extend(n for n in row["cast_list"] if n.lower() in q_lower)
    if row["director"] and row["director"].lower() in q_lower:
        reasons.append(row["director"])

    matched_keywords = [k for k in row["keywords_list"]
                        if any(word in query_tokens for word in k.lower().split())][:4]
    reasons.extend(matched_keywords)

    overview_tokens = set(row["overview_clean"].split())
    reasons.extend(sorted(query_tokens & overview_tokens)[:4])

    seen, unique_reasons = set(), []
    for r in reasons:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique_reasons.append(r)
    return unique_reasons[:6]


def breakdown_html(row: pd.Series) -> str:
    """Level 2 only: how much each feature contributed to this movie's score."""
    if "c_plot" not in row.index or row["similarity"] <= 0:
        return ""
    parts = []
    for f in FEATURES:
        share = row[f"c_{f}"] / row["similarity"] * 100
        if share >= 1:
            parts.append(f'<span class="score-chip">{FEATURE_LABELS[f]} {share:.0f}%</span>')
    if not parts:
        return ""
    return f'<div><span class="breakdown-label">Score made of</span>{"".join(parts)}</div>'


def match_gauge_svg(pct: float) -> str:
    """A small circular gold gauge showing relative relevance, film-reel style."""
    pct = max(0.0, min(100.0, float(pct)))
    radius = 30
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct / 100)
    html = f"""<div class="match-gauge">
<svg width="76" height="76" viewBox="0 0 76 76">
<circle cx="38" cy="38" r="{radius}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="6"/>
<circle cx="38" cy="38" r="{radius}" fill="none" stroke="#E3B23C" stroke-width="6" stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}" transform="rotate(-90 38 38)"/>
<text x="38" y="43" text-anchor="middle" class="pct">{pct:.0f}%</text>
</svg>
<div class="label">RELEVANCE</div>
</div>"""
    return html


def render_ticket_card(row: pd.Series, query: str) -> None:
    year = str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "—"
    genre_pills = "".join(f'<span class="ticket-pill">{g}</span>' for g in row["genres_list"][:4])
    reasons = why_recommended(query, row)
    reason_chips = "".join(f'<span class="reason-chip">{r}</span>' for r in reasons)
    because_html = (
        f'<div><span class="because-label">Because you mentioned</span>{reason_chips}</div>'
        if reasons else ""
    )
    director_html = (
        f'<span class="ticket-year-rating">🎬 {row["director"]}</span>' if row["director"] else ""
    )

    card_html = (
        f'<div class="ticket-card">'
        f'<div style="flex: 1;">'
        f'<div class="ticket-title">{row["title"]}</div>'
        f'<div class="ticket-meta">'
        f'<span class="ticket-year-rating">{year}</span>'
        f'<span class="ticket-year-rating">⭐ {row["vote_average"]}</span>'
        f'{director_html}'
        f'{genre_pills}'
        f'</div>'
        f'<div class="ticket-overview">{row["overview"]}</div>'
        f'{because_html}'
        f'{breakdown_html(row)}'
        f'</div>'
        f'{match_gauge_svg(row["similarity_%"])}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="cinema-hero"><h1>CINEMATCH</h1>'
    "<p>Describe the movie you're in the mood for — the projector does the rest.</p></div>",
    unsafe_allow_html=True,
)

col_search, col_n = st.columns([4, 1])
with col_search:
    query = st.text_input(
        "Query",
        placeholder="e.g. science fiction space adventure with an emotional story",
        label_visibility="collapsed",
    )
with col_n:
    top_n = st.number_input("Picks", min_value=1, max_value=20, value=5, step=1, label_visibility="collapsed")

mode = st.radio("Engine", [WEIGHTED, BASIC], horizontal=True,
                help="Weighted blends plot, genre, keywords, cast and director. "
                     "Basic is the Level 1 baseline (one merged TF-IDF).")

with st.expander("🎟️ Refine your pick"):
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        genre_choice = st.selectbox("Genre", ["All"] + all_genres)
    with fcol2:
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)

    user_weights = dict(saved_weights)
    if mode == WEIGHTED:
        st.markdown("**Feature weights** (auto-normalised to 100%; cast/director only count "
                    "when you name someone in the query)")
        wcols = st.columns(5)
        for col, f in zip(wcols, FEATURES):
            with col:
                user_weights[f] = st.slider(FEATURE_LABELS[f], 0.0, 1.0,
                                            float(saved_weights.get(f, DEFAULT_WEIGHTS[f])), 0.05,
                                            key=f"w_{f}")

search_clicked = st.button("🎬  Roll the film", type="primary")

if search_clicked or query:
    if not query.strip():
        st.warning("Type something first, e.g. 'funny movie about friendship and college life'.")
    else:
        results = recommend(query, mode=mode, top_n=int(top_n), genre_filter=genre_choice,
                            min_rating=min_rating, weights=user_weights)

        if results.empty:
            st.info("Nothing in the current listings — try loosening the filters.")
        else:
            st.markdown(f'<div class="now-showing">NOW SHOWING — {len(results)} PICKS</div>', unsafe_allow_html=True)
            for _, row in results.iterrows():
                render_ticket_card(row, query)
else:
    st.markdown(
        '<p style="color:#9C9289; text-align:center; margin-top:2rem;">'
        "The marquee's empty — enter a query above and press <b>Roll the film</b>.</p>",
        unsafe_allow_html=True,
    )
