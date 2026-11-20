import json
import random
from pathlib import Path
import gensim.downloader as api
import numpy as np
import pandas as pd
import streamlit as st
from wordfreq import top_n_list, zipf_frequency
from nltk.corpus import wordnet as wn


st.set_page_config(page_title="Contexto — testiranje modela", layout="wide")

MODEL_NAMES = {
    "GloVe": "glove-wiki-gigaword-50",
    "Word2Vec": "word2vec-google-news-300",
}

@st.cache_resource
def load_model(name):
    return api.load(name)

@st.cache_data
def get_vocabulary():
    glove = load_model(MODEL_NAMES["GloVe"])
    word2vec = load_model(MODEL_NAMES["Word2Vec"])

    return tuple(
        word for word in top_n_list("en", 20000)
        if word.isalpha()
        and word.isascii()
        and 2 <= len(word) <= 15
        and word in glove
        and word in word2vec
    )

@st.cache_data
def get_target_words(words):
    file_path = Path(__file__).with_name("target_words.json")

    with open(file_path, "r", encoding="utf-8") as file:
        candidate_words = json.load(file)

    vocabulary = set(words)

    concrete_categories = {
        "noun.animal",
        "noun.artifact",
        "noun.body",
        "noun.food",
        "noun.object",
        "noun.plant",
        "noun.substance",
        "noun.person",
        "noun.group",
        "noun.phenomenon",
    }

    targets = []

    for word in candidate_words:
        word = word.strip().lower()

        if (
            not word
            or not word.isascii()
            or not word.isalpha()
            or len(word) < 3
            or word not in vocabulary
            or zipf_frequency(word, "en") < 3.5
        ):
            continue

        noun_synsets = wn.synsets(word, pos=wn.NOUN)

        if not any(
                synset.lexname() in concrete_categories
                for synset in noun_synsets
        ):
            continue

        targets.append(word)

    return tuple(dict.fromkeys(targets))


@st.cache_data
def calculate_ranks(model_name, target_word, words):
    model = load_model(model_name)
    indices = [model.key_to_index[word] for word in words]
    vectors = model.vectors[indices]
    target_vector = model[target_word]

    similarities = ( vectors @ target_vector ) / ( np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector) )

    order = np.argsort(-similarities, kind="stable")

    ranks = {
        words[int(index)]: rank
        for rank, index in enumerate(order, start=1)
    }
    similarity_dict = {
        words[index]: float(similarities[index])
        for index in range(len(words))
    }

    return ranks, similarity_dict


def render_ranked_words(ranks, similarities, top_n, model_label, model_color):
    ranked_words = sorted(ranks, key=ranks.get)[:top_n]

    st.markdown(
        f'<h3 style="color:{model_color}; margin-bottom:0.25rem;">'
        f'{model_label}</h3>',
        unsafe_allow_html=True,
    )

    header1, header2, header3 = st.columns([3, 1, 1])
    header2.markdown("**Rang**")
    header3.markdown("**Sličnost**")

    for word in ranked_words:
        rank = ranks[word]
        similarity = similarities[word]

        col_word, col_rank, col_similarity = st.columns([3, 1, 1])

        col_word.markdown(f"**{word}**")
        col_rank.markdown(f"#{rank:,}")
        col_similarity.caption(f"{similarity:.4f}")


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">contexto</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Testiranje i poređenje GloVe i Word2Vec modela</div>',
    unsafe_allow_html=True,
)

with st.spinner("Učitavanje modela i priprema zajedničkog vokabulara..."):
    vocabulary = get_vocabulary()

if not vocabulary:
    st.error("Zajednički vokabular je prazan.")
    st.stop()

vocabulary_set = set(vocabulary)
target_words = get_target_words(vocabulary)

if "analysis_target" not in st.session_state:
    preferred_word = "ocean"
    if target_words:
        st.session_state.analysis_target = (
            preferred_word if preferred_word in target_words else target_words[0]
        )
    else:
        st.session_state.analysis_target = (
            preferred_word if preferred_word in vocabulary_set else vocabulary[0]
        )


if "target_input" not in st.session_state:
    st.session_state.target_input = st.session_state.analysis_target


def choose_random_word():
    candidates = [
        word for word in (target_words or vocabulary)
        if word != st.session_state.analysis_target
    ]

    if candidates:
        word = random.choice(candidates)
        st.session_state.analysis_target = word
        st.session_state.target_input = word
        st.session_state.analysis_ready = True

with st.container(border=True):
    st.markdown("#### Podešavanja analize")

    col_target, col_count = st.columns([3, 1.2])

    with col_target:
        target_input = st.text_input(
            "Ciljna riječ (engleski)",
            key="target_input",
        ).strip().lower()

    with col_count:
        top_n = st.selectbox(
            "Broj riječi",
            options=[10, 20, 30, 50, 100],
            index=1,
        )

    col_random, col_analyze = st.columns(2)

    with col_random:
        st.button(
            "Nasumična riječ",
            use_container_width=True,
            on_click=choose_random_word,
        )

    with col_analyze:
        analyze_clicked = st.button(
            "Analiziraj",
            type="primary",
            use_container_width=True,
        )

if analyze_clicked:
    if not target_input:
        st.error("Unesi ciljnu riječ.")
        st.stop()

    if len(target_input.split()) != 1:
        st.error("Unesi samo jednu riječ.")
        st.stop()

    if target_input not in vocabulary_set:
        st.error("Ta riječ nije u zajedničkom vokabularu oba modela.")
        st.stop()

    st.session_state.analysis_target = target_input
    st.session_state.analysis_ready = True


if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = True

st.caption(
    f"Zajednički vokabular: {len(vocabulary):,} riječi. "
    + (
        f"Mogućih tajnih riječi prema pravilima igrice: {len(target_words):,}."
        if target_words
        else "Fajl target_words.json nije pronađen ili ne sadrži važeće riječi."
    )
)

target_word = st.session_state.analysis_target

if target_word not in vocabulary_set:
    st.warning(
        "Trenutna ciljna riječ nije u zajedničkom vokabularu. "
        "Unesi važeću englesku riječ i klikni „Analiziraj“."
    )
    st.stop()

if st.session_state.analysis_ready:
    with st.spinner(f"Računanje rangova za „{target_word}“..."):
        glove_ranks, glove_similarities = calculate_ranks(
            MODEL_NAMES["GloVe"], target_word, vocabulary
        )
        word2vec_ranks, word2vec_similarities = calculate_ranks(
            MODEL_NAMES["Word2Vec"], target_word, vocabulary
        )

    st.divider()
    st.subheader(f"{target_word}")
    left, right = st.columns(2, gap="large")

    with left:
        render_ranked_words(
            glove_ranks,
            glove_similarities,
            top_n,
            "GloVe",
            "#3978B8",
        )

    with right:
        render_ranked_words(
            word2vec_ranks,
            word2vec_similarities,
            top_n,
            "Word2Vec",
            "#E5A044",
        )


    st.divider()
    st.subheader("Najveća odstupanja između modela")

    glove_top100 = set(
        sorted(glove_ranks, key=glove_ranks.get)[:100]
    )
    word2vec_top100 = set(
        sorted(word2vec_ranks, key=word2vec_ranks.get)[:100]
    )

    # Unija - rijeci koje su u prvih 100 bar jednog modela
    candidate_words = glove_top100 | word2vec_top100

    comparison = pd.DataFrame([
        {
            "Riječ": word,
            "GloVe rang": glove_ranks[word],
            "Word2Vec rang": word2vec_ranks[word],
            "Odstupanje": abs(
                glove_ranks[word] - word2vec_ranks[word]
            ),
        }
        for word in candidate_words
    ]).sort_values(
        "Odstupanje", ascending=False
    ).reset_index(drop=True)

    if not comparison.empty:
        st.caption(
            "Porede se riječi iz prvih 100 rezultata bilo kog modela; "
            "rang u drugom modelu može biti i veći od 100."
        )

        st.dataframe(
            comparison.head(10),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Provjeri rang konkretnih riječi")
    st.caption(
        "Unesi riječi razdvojene zarezom ili novim redom: "
    )
    words_text = st.text_area(
        "",
        placeholder="sea, beach, water, fish",
        key="words_to_check",
    )

    if st.button("Uporedi"):
        requested_words = [
            word.strip().lower()
            for word in words_text.replace("\n", ",").split(",")
            if word.strip()
        ]
        requested_words = list(dict.fromkeys(requested_words))

        if not requested_words:
            st.warning("Unesi bar jednu riječ.")
        else:
            valid_words = [
                word for word in requested_words if word in vocabulary_set
            ]
            invalid_words = [
                word for word in requested_words if word not in vocabulary_set
            ]

            if invalid_words:
                st.warning(
                    "Nisu u zajedničkom vokabularu: "
                    + ", ".join(invalid_words)
                )

            if valid_words:
                comparison = pd.DataFrame(
                    [
                        {
                            "Riječ": word,
                            "GloVe rang": glove_ranks[word],
                            "GloVe sličnost": glove_similarities[word],
                            "Word2Vec rang": word2vec_ranks[word],
                            "Word2Vec sličnost": word2vec_similarities[word],
                            "Razlika rangova": abs(
                                glove_ranks[word] - word2vec_ranks[word]
                            ),
                            "Bolje rangiran model": (
                                "GloVe"
                                if glove_ranks[word] < word2vec_ranks[word]
                                else "Word2Vec"
                                if word2vec_ranks[word] < glove_ranks[word]
                                else "Isto"
                            ),
                        }
                        for word in valid_words
                    ]
                ).sort_values("GloVe rang")

                st.dataframe(
                    comparison.round(4),
                    hide_index=True,
                    use_container_width=True,
                )