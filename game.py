import random
import numpy as np
import streamlit as st
import gensim.downloader as api
from wordfreq import top_n_list, zipf_frequency
from nltk.corpus import wordnet as wn
import json
from pathlib import Path


st.set_page_config(page_title="Contexto", layout="centered")

MODEL_NAMES = {
    "GloVe": "glove-wiki-gigaword-50",
    "Word2Vec": "word2vec-google-news-300"
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

    # Uklanjanje duplikata
    return tuple(dict.fromkeys(targets))


@st.cache_data
def calculate_ranks(model_name, target_word, words):
    model = load_model(model_name)

    indices = [model.key_to_index[w] for w in words]
    vectors = model.vectors[indices]
    target_vector = model[target_word]

    similarities = ( vectors @ target_vector ) / ( np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector) )

    order = np.argsort(-similarities, kind="stable")

    return {
        words[int(i)]: rank
        for rank, i in enumerate(order, start=1)
    }


st.markdown("""
<style>
.block-container {
    max-width: 700px;
    padding-top: 2rem;
}
.game-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0;
}
.guess-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 13px 0 4px 0;
}
.guess-row-new {
    background-color: #e8f5e9;
    border-radius: 6px;
}
.guess-row-repeated {
    background-color: #fff3cd;
    border-radius: 6px;
}
.guess-word {
    width: 115px;
    overflow-wrap: anywhere;
    font-weight: 600;
}
.guess-rank {
    width: 75px;
    text-align: right;
    font-weight: 600;
}
.bar-bg {
    flex: 1;
    height: 18px;
    background: #e5e7eb;
    border-radius: 5px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 5px;
}
div[data-testid="stFormSubmitButton"] {
    display: none;
}
.game-message {
    padding: 8px 12px;
    margin: 8px 0 12px 0;
    border-radius: 6px;
    font-size: 0.85rem;
}
.game-message-info {
    background-color: #e8f0fe;
    color: #244780;
}
.game-message-warning {
    background-color: #fff3cd;
    color: #795600;
}
.game-message-success {
    background-color: #e8f5e9;
    color: #216e39;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="game-title">contexto</div>',
    unsafe_allow_html=True
)


model_choice = st.selectbox("Model", list(MODEL_NAMES.keys()))

with st.spinner("Loading models..."):
    model = load_model(MODEL_NAMES[model_choice])
    words = get_vocabulary()
    target_words = get_target_words(words)

word_set = set(words)

if len(target_words) < 2:
    st.error("There are not enough words to play.")
    st.stop()


def new_word(previous=None):
    choices = [w for w in target_words if w != previous]
    return random.choice(choices)


if "guesses" not in st.session_state:
    st.session_state.guesses = []

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "gave_up" not in st.session_state:
    st.session_state.gave_up = False

if "hints_given" not in st.session_state:
    st.session_state.hints_given = []

if "repeated_guess" not in st.session_state:
    st.session_state.repeated_guess = None

if "last_guess" not in st.session_state:
    st.session_state.last_guess = None

if "game_message" not in st.session_state:
    st.session_state.game_message = None

if (
    "target_word" not in st.session_state
    or st.session_state.target_word not in target_words
):
    st.session_state.target_word = new_word()
    st.session_state.guesses = []
    st.session_state.game_over = False
    st.session_state.gave_up = False
    st.session_state.hints_given = []
    st.session_state.repeated_guess = None
    st.session_state.game_message = None


target_word = st.session_state.target_word

# test
# st.write(f"**Secret Word**: {target_word}")

with st.spinner("Preparing the game..."):
    ranks = calculate_ranks(
        MODEL_NAMES[model_choice],
        target_word,
        words
    )

# Hint i odustajanje
col1, col2 = st.columns(2)

with col1:
    if st.button(
        "Hint",
        use_container_width=True,
        disabled=st.session_state.game_over
    ):
        candidates = [
            word for word in sorted(ranks, key=ranks.get)
            if word != target_word
            and word not in st.session_state.guesses
            and word not in st.session_state.hints_given
            and 2 <= ranks[word] <= 20
        ]

        if candidates:
            hint_word = random.choice(candidates)
            st.session_state.hints_given.append(hint_word)
            st.session_state.game_message = {
                "type": "info",
                "text": f"Hint: Try a word related to '{hint_word}'."
            }
        else:
            st.session_state.game_message = {
                "type": "info",
                "text": "No more hints available."
            }

with col2:
    if st.button(
        "Give up",
        use_container_width=True,
        disabled=st.session_state.game_over
    ):
        st.session_state.gave_up = True
        st.session_state.game_over = True
        st.rerun()

# Unos rijeci
with st.form("guess_form", clear_on_submit=True, border=False):
    guess = st.text_input(
        " ",
        placeholder="Enter a word...",
        label_visibility="collapsed",
        disabled=st.session_state.game_over
    )

    submitted = st.form_submit_button("Guess")

if submitted and not st.session_state.game_over:
    guess = guess.strip().lower()
    st.session_state.game_message = None

    if not guess:
        st.session_state.game_message = {
            "type": "warning",
            "text": "Please enter a word."
        }

    elif len(guess.split()) != 1:
        st.session_state.game_message = {
            "type": "warning",
            "text": "Please enter a single word."
        }

    elif guess not in word_set:
        st.session_state.game_message = {
            "type": "warning",
            "text": "This word is not available in the game."
        }

    elif guess in st.session_state.guesses:
        st.session_state.last_guess = None
        st.session_state.repeated_guess = guess
        st.session_state.game_message = {
            "type": "info",
            "text": f"You've already guessed '{guess}'."
        }

    else:
        st.session_state.repeated_guess = None
        st.session_state.last_guess = guess
        st.session_state.guesses.append(guess)

        if guess == target_word:
            st.session_state.game_over = True
            st.session_state.game_message = {
                "type": "success",
                "text": f"You guessed it! The secret word was {target_word}."
            }

        st.rerun()

if st.session_state.gave_up:
    st.markdown(
        f'<div class="game-message game-message-info">'
        f'The secret word was: <b>{target_word}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

elif st.session_state.game_message is not None:
    message = st.session_state.game_message

    message_class = {
        "info": "game-message-info",
        "warning": "game-message-warning",
        "success": "game-message-success"
    }.get(message["type"], "game-message-info")

    st.markdown(
        f'<div class="game-message {message_class}">'
        f'{message["text"]}'
        f'</div>',
        unsafe_allow_html=True
    )

st.write(f"**Guesses: {len(st.session_state.guesses)}**")

# Lista pokusaja
sorted_guesses = sorted(
    st.session_state.guesses,
    key=lambda word: ranks[word]
)

for word in sorted_guesses:
    rank = ranks[word]

    if st.session_state.repeated_guess is not None:
        if word == st.session_state.repeated_guess:
            row_class = "guess-row-repeated"
        else:
            row_class = ""

    elif word == st.session_state.last_guess:
        row_class = "guess-row-new"

    else:
        row_class = ""

    progress = max(0.02, 1 - np.log(rank) / np.log(len(words)))

    if rank <= 100:
        color = "#16a085"
    elif rank <= 1000:
        color = "#8bc34a"
    elif rank <= 3000:
        color = "#f1c40f"
    elif rank <= 7000:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    st.markdown(
        f"""
        <div class="guess-row {row_class}">
            <div class="guess-word">{word}</div>
            <div class="bar-bg">
                <div class="bar-fill"
                     style="width:{progress * 100:.2f}%;
                            background:{color};">
                </div>
            </div>
            <div class="guess-rank">{rank:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


if st.button("New game", use_container_width=True):
    st.session_state.target_word = new_word(target_word)
    st.session_state.guesses = []
    st.session_state.game_over = False
    st.session_state.gave_up = False
    st.session_state.hints_given = []
    st.session_state.repeated_guess = None
    st.session_state.last_guess = None
    st.session_state.game_message = None
    st.rerun()