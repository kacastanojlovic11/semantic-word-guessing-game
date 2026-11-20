# Semantic Word Guessing Game — GloVe vs. Word2Vec

A Python/Streamlit project inspired by semantic word-guessing games. Players guess a hidden English word and receive feedback based on the cosine similarity of pretrained word embeddings. An accompanying analysis app compares the rankings produced by **GloVe** and **Word2Vec**.

> This is an independent educational project inspired by the general idea of Contexto. It is not affiliated with or endorsed by Contexto.

## Main features

- **Game (`game.py`)**: Select GloVe or Word2Vec, enter guesses, view their semantic ranks, request hints, give up, or start another game.
- **Model comparison (`main.py`)**: Choose a target word; inspect the 10, 20, 30, 50, or 100 nearest words for each model; examine rank differences; and compare specific guesses.
- **Exploratory evaluation (`evaluation.py`)**: Compare the ranks of hand-selected related terms and unrelated control terms across 25 target words. Five polysemous words are explored separately.

## Technologies

Python, Streamlit, NumPy, pandas, Gensim, wordfreq, and NLTK WordNet.

Pretrained embeddings:

- `glove-wiki-gigaword-50` (50-dimensional vectors)
- `word2vec-google-news-300` (300-dimensional vectors)

No embedding models are trained by this application.

## Method

1. Build a shared English vocabulary from the 20,000 most frequent entries supplied by `wordfreq`, filtering for ASCII alphabetic words, length 2–15, and presence in both pretrained models.
2. Load candidate target words from `target_words.json`, then apply frequency and WordNet noun-category filters.
3. Compute cosine similarity between the target vector and vectors in the shared vocabulary.
4. Sort the scores in descending order to obtain each guess's rank. The same shared vocabulary makes the two ranking outputs directly comparable.

The evaluation is **exploratory**: its examples are hand-selected and its ordering scores do not measure general model accuracy. Some related terms are antonyms, so this evaluation also measures contextual association rather than synonymy alone.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Run one application at a time:

```bash
streamlit run game.py
streamlit run main.py
streamlit run evaluation.py
```

**Important:** The first execution may download large pretrained models. The `word2vec-google-news-300` archive is approximately 1.74 GB and loading it can require substantially more RAM. The model files are **not** included in this repository. An internet connection and sufficient disk space/memory are required for the initial model download.

The file `target_words.json` is required next to `game.py` and `main.py`.

## Repository contents

```text
semantic-word-guessing-game/
├── game.py
├── main.py
├── evaluation.py
├── target_words.json
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/              # Optional
    ├── game.png
    └── comparison.png
```

## Example screenshots

Add screenshots of the game and model comparison interface here before publishing.

## Notes

- Both models are loaded through Gensim's data downloader.
- Model redistribution and usage are subject to the original providers' terms.
- Exact dependency versions can be recorded in `requirements.txt` after confirming a clean installation on your machine.
