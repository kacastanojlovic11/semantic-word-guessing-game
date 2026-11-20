"""Reproducible comparison of pretrained GloVe and Word2Vec embeddings.
Run: streamlit run evaluation.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import gensim.downloader as api
from wordfreq import top_n_list

import plotly.graph_objects as go

MODEL_NAMES = {
    "GloVe": "glove-wiki-gigaword-50",
    "Word2Vec": "word2vec-google-news-300",
}

# First five words explore polysemy; the remaining groups use the first
# three guesses as related and the fourth as an intentionally unrelated control.
TEST_CASES = {
    "apple": ["pear", "fruit", "computer", "banana"],
    "mouse": ["rat", "keyboard", "cheese", "animal"],
    "bank": ["money", "river", "loan", "finance"],
    "spring": ["summer", "water", "winter", "flower"],
    "bat": ["animal", "baseball", "bird", "sport"],
    "happy": ["joyful", "glad", "sad", "computer"],
    "beautiful": ["pretty", "attractive", "ugly", "river"],
    "smart": ["intelligent", "clever", "stupid", "banana"],
    "big": ["large", "huge", "small", "piano"],
    "fast": ["quick", "rapid", "slow", "mountain"],
    "dog": ["cat", "puppy", "animal", "computer"],
    "car": ["truck", "vehicle", "bus", "flower"],
    "fruit": ["apple", "banana", "orange", "keyboard"],
    "color": ["red", "blue", "green", "teacher"],
    "music": ["piano", "guitar", "melody", "finance"],
    "school": ["teacher", "student", "classroom", "ocean"],
    "doctor": ["hospital", "patient", "medicine", "guitar"],
    "ocean": ["sea", "water", "fish", "computer"],
    "football": ["soccer", "player", "stadium", "flower"],
    "king": ["queen", "prince", "castle", "keyboard"],
    "computer": ["software", "keyboard", "internet", "forest"],
    "book": ["reading", "author", "library", "banana"],
    "city": ["town", "street", "building", "guitar"],
    "food": ["meal", "restaurant", "cooking", "river"],
    "nature": ["forest", "mountain", "environment", "computer"],
}
POLYSEMY_WORDS = {"apple", "mouse", "bank", "spring", "bat"}

@st.cache_resource
def load_model(name):
    return api.load(name)

@st.cache_data
def get_vocabulary():
    glove = load_model(MODEL_NAMES["GloVe"])
    word2vec = load_model(MODEL_NAMES["Word2Vec"])
    return tuple(
        word for word in top_n_list("en", 20000)
        if word.isalpha() and word.isascii() and 2 <= len(word) <= 15
        and word in glove and word in word2vec
    )

@st.cache_data
def calculate_scores(model_name, target, vocabulary):
    model = load_model(model_name)
    idx = [model.key_to_index[word] for word in vocabulary]
    vectors = model.vectors[idx]
    target_vector = model[target]
    similarities = vectors @ target_vector / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector)
    )
    order = np.argsort(-similarities, kind="stable")
    ranks = {vocabulary[int(i)]: rank for rank, i in enumerate(order, start=1)}
    return ranks


@st.cache_data
def run_evaluation(cases, vocabulary):
    models = {name: load_model(identifier) for name, identifier in MODEL_NAMES.items()}
    vocab_set = set(vocabulary)
    rows = []
    for target, guesses in cases:
        missing_target = target not in vocab_set
        ranks = (
            {name: calculate_scores(identifier, target, vocabulary)
             for name, identifier in MODEL_NAMES.items()}
            if not missing_target else {}
        )
        for guess in guesses:
            row = {"Secret word": target, "Guess": guess}
            valid = not missing_target and guess in vocab_set
            for name, model in models.items():
                row[f"{name} similarity"] = round(float(model.similarity(target, guess)), 4) if valid else np.nan
                row[f"{name} rank"] = ranks[name][guess] if valid else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_pairwise_order(df, cases):
    """Compare three prespecified related guesses to an unrelated control."""
    comparisons = []
    skipped = []
    for target, guesses in cases.items():
        if target in POLYSEMY_WORDS:
            continue
        if len(guesses) != 4:
            skipped.append((target, "Expected exactly four guesses"))
            continue
        group = df[df["Secret word"] == target].set_index("Guess")
        related, control = guesses[:3], guesses[3]
        for model_name in MODEL_NAMES:
            rank_col = f"{model_name} rank"
            for word in related:
                if (word not in group.index or control not in group.index
                        or pd.isna(group.loc[word, rank_col])
                        or pd.isna(group.loc[control, rank_col])):
                    skipped.append((target, model_name, word, control))
                    continue
                related_rank = int(group.loc[word, rank_col])
                control_rank = int(group.loc[control, rank_col])
                comparisons.append({
                    "Model": model_name,
                    "Secret word": target,
                    "Related word": word,
                    "Control word": control,
                    "Related rank": related_rank,
                    "Control rank": control_rank,
                    "Correct order": related_rank < control_rank,
                })
    detail = pd.DataFrame(comparisons)
    if detail.empty:
        summary = pd.DataFrame(columns=["Model", "Correct", "Total", "Success rate (%)"])
    else:
        summary = (detail.groupby("Model", sort=False)["Correct order"]
                   .agg(Correct="sum", Total="count").reset_index())
        summary["Success rate (%)"] = (100 * summary["Correct"] / summary["Total"]).round(1)
    return summary, detail, skipped

st.title("NLP Model Evaluation")
st.write("GloVe versus Word2Vec on a shared, filtered English vocabulary.")
st.caption("Rank 1 is the closest word; lower ranks are better. The results are exploratory, not a benchmark of general model accuracy.")

with st.expander("Methodology and limitations"):
    st.write("Vocabulary: common English words from wordfreq's first 20,000 entries, filtered to ASCII letters, lengths 2–15, and membership in both pretrained models. The same ranking vocabulary is used for both models.")
    st.write("For 20 target words, the first three guesses are considered related and the fourth an unrelated control. The five polysemous target words are analyzed separately. Some 'related' words are antonyms: the metric tests contextual association rather than synonymy.")

if st.button("Run Evaluation"):
    with st.spinner("Loading models and calculating results..."):
        vocabulary = get_vocabulary()
        results = run_evaluation(tuple((k, tuple(v)) for k, v in TEST_CASES.items()), vocabulary)
        summary, details, skipped = evaluate_pairwise_order(results, TEST_CASES)
    st.session_state.eval_results = results
    st.session_state.eval_summary = summary
    st.session_state.eval_details = details
    st.session_state.eval_skipped = skipped
    st.session_state.eval_vocab_size = len(vocabulary)

if "eval_results" in st.session_state:
    results = st.session_state.eval_results
    summary = st.session_state.eval_summary
    details = st.session_state.eval_details
    missing = int(results[["GloVe rank", "Word2Vec rank"]].isna().any(axis=1).sum())
    a, b, c, d = st.columns(4)
    a.metric("Target words", results["Secret word"].nunique())
    b.metric("Test pairs", len(results))
    c.metric("Missing pairs", missing)
    d.metric("Ranking vocabulary", f"{st.session_state.eval_vocab_size:,}")

    st.subheader("Pairwise ordering accuracy")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("A success means a preselected related term has a smaller rank than the unrelated control term. This is a descriptive score for this hand-picked test set only.")

    errors = details[~details["Correct order"]]
    st.subheader("Exceptions")
    if errors.empty:
        st.success("No ordering exceptions among valid comparisons.")
    else:
        st.dataframe(errors, hide_index=True, width="stretch")
    if st.session_state.eval_skipped:
        st.warning(f"Skipped comparisons: {len(st.session_state.eval_skipped)}")

    st.subheader("All model results")
    st.dataframe(results, hide_index=True, width="stretch")
    target = st.selectbox("Target word for chart", list(TEST_CASES))
    plot_data = results.loc[results["Secret word"] == target].set_index("Guess")
    # st.subheader("Similarity comparison")
    # st.bar_chart(plot_data[["GloVe similarity", "Word2Vec similarity"]])
    # st.bar_chart(plot_data[["GloVe rank", "Word2Vec rank"]])

    st.subheader("1. Direktno poređenje rangova — horizontalni grafikon")

    rank_data = (
        plot_data[["GloVe rank", "Word2Vec rank"]]
        .dropna()
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=rank_data["Guess"],
        x=rank_data["GloVe rank"],
        name="GloVe",
        orientation="h",
        marker_color="#3978B8",
        text=rank_data["GloVe rank"].astype(int).astype(str),
        textposition="outside",
        hovertemplate="Riječ: %{y}<br>GloVe rang: %{x}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=rank_data["Guess"],
        x=rank_data["Word2Vec rank"],
        name="Word2Vec",
        orientation="h",
        marker_color="#E5A044",
        text=rank_data["Word2Vec rank"].astype(int).astype(str),
        textposition="outside",
        hovertemplate="Riječ: %{y}<br>Word2Vec rang: %{x}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        height=350,
        xaxis=dict(
            title="Rang (logaritamska skala, niže = bolje)",
            type="log",
            rangemode="tozero",
        ),
        yaxis_title="",
        legend_title="Model",
        template="plotly_white",
        margin=dict(l=20, r=60, t=30, b=50),
    )

    st.plotly_chart(fig, width="stretch")

    st.download_button("Download results CSV", results.to_csv(index=False).encode("utf-8"), "model_evaluation.csv", "text/csv")
    st.download_button("Download summary CSV", summary.to_csv(index=False).encode("utf-8"), "evaluation_summary.csv", "text/csv")
    st.download_button("Download comparison details CSV", details.to_csv(index=False).encode("utf-8"), "evaluation_comparisons.csv", "text/csv")
