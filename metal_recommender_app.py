
import streamlit as st
import pandas as pd
import numpy as np
import re
from numpy.linalg import norm

st.set_page_config(page_title="Psycho social Metal 🤘 Analysis", page_icon="🤘", layout="wide")

st.title("Psycho-social Metal 🤘 Analysis")
st.caption("Questionnaire → trait vector → top 5 metal subgenres + band examples")

TRAIT_NAMES = ["Openness","Nonconformity/Uniqueness","Authority skepticism",
               "Extraversion","Emotion regulation via music","Sensation-seeking"]

genre_map = {
    "black metal":              [0.70,0.80,0.75,0.30,0.60,0.75],
    "atmospheric black metal":  [0.75,0.80,0.70,0.35,0.60,0.70],
    "post-black metal":         [0.75,0.75,0.60,0.40,0.60,0.70],
    "melodic black metal":      [0.72,0.78,0.70,0.35,0.60,0.72],
    "death metal":              [0.65,0.70,0.65,0.35,0.65,0.75],
    "technical death metal":    [0.78,0.70,0.60,0.35,0.62,0.75],
    "brutal death metal":       [0.60,0.72,0.65,0.30,0.62,0.80],
    "melodic death metal":      [0.70,0.70,0.60,0.40,0.65,0.70],
    "blackened death metal":    [0.70,0.75,0.70,0.30,0.65,0.75],
    "doom metal":               [0.65,0.60,0.45,0.35,0.60,0.50],
    "death-doom":               [0.66,0.65,0.50,0.33,0.62,0.55],
    "funeral doom":             [0.68,0.68,0.50,0.30,0.62,0.50],
    "stoner doom":              [0.66,0.62,0.45,0.45,0.60,0.55],
    "thrash metal":             [0.70,0.60,0.60,0.40,0.60,0.70],
    "groove metal":             [0.60,0.60,0.55,0.40,0.60,0.60],
    "speed metal":              [0.66,0.55,0.45,0.55,0.55,0.65],
    "heavy metal":              [0.55,0.40,0.35,0.60,0.50,0.50],
    "nwobhm":                   [0.60,0.50,0.40,0.55,0.50,0.60],
    "power metal":              [0.65,0.45,0.30,0.70,0.50,0.55],
    "symphonic metal":          [0.70,0.50,0.30,0.60,0.55,0.50],
    "folk metal":               [0.65,0.55,0.35,0.60,0.55,0.55],
    "progressive metal":        [0.85,0.60,0.50,0.40,0.60,0.60],
    "djent":                    [0.82,0.62,0.52,0.38,0.58,0.60],
    "avant-garde metal":        [0.86,0.70,0.55,0.38,0.60,0.60],
    "industrial metal":         [0.60,0.60,0.60,0.40,0.55,0.60],
    "gothic metal":             [0.66,0.58,0.45,0.45,0.60,0.55],
    "alternative metal":        [0.60,0.58,0.50,0.50,0.60,0.60],
    "nu metal":                 [0.55,0.60,0.50,0.50,0.60,0.60],
    "metalcore":                [0.60,0.60,0.50,0.50,0.60,0.65],
    "deathcore":                [0.60,0.68,0.58,0.40,0.60,0.72],
    "post-metal":               [0.74,0.66,0.55,0.45,0.62,0.60],
    "sludge metal":             [0.64,0.62,0.50,0.40,0.60,0.58],
}

GENRE_KEYS = sorted(genre_map.keys(), key=len, reverse=True)

# levenstein? ali cosine?
def cosine(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def normalize_genre_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[(){}\[\];:!?.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data
def load_bands(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    #bands_df = pd.read_csv("demo_bands.csv")

    if "Band" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name":"Band"})
    if "Genre" not in df.columns:
        if "genre" in df.columns:
            df = df.rename(columns={"genre":"Genre"})
    if "Country" not in df.columns and "country" in df.columns:
        df = df.rename(columns={"country":"Country"})

    if "Genre" in df.columns:
        df["Genre_norm"] = df["Genre"].astype(str).map(normalize_genre_text)
    else:
        df["Genre_norm"] = ""
    return df



# Sidebar for CSV
#st.sidebar.header("Band dataset")
#bands_df = pd.read_csv("demo_bands.csv")


# Questionnaire
st.subheader("1) Questionnaire (rate 1–5)")
QUESTIONS = {
    "Openness": [
        "I enjoy discovering new or experimental music styles.",
        "I appreciate complex or unconventional art.",
        "I like music that challenges mainstream trends.",
        "I’m curious to learn about different subcultures.",
        "Creativity and imagination are important to me."
    ],
    "Nonconformity/Uniqueness": [
        "I prefer to stand out rather than fit in.",
        "I’m comfortable liking things most people dislike.",
        "I enjoy expressing individuality through music/fashion.",
        "Rules and conventions often feel restrictive to me.",
        "I value authenticity more than popularity."
    ],
    "Authority skepticism": [
        "I question traditional institutions and authority.",
        "People should think for themselves.",
        "I dislike being told what to do or believe.",
        "Rebellion can be healthy for society.",
        "I relate to anti-establishment messages in art/music."
    ],
    "Extraversion": [
        "I enjoy social gatherings and meeting new people.",
        "I feel energized when I’m around others.",
        "I like being in large, loud, energetic concerts.",
        "I often seek out group experiences rather than solo ones.",
        "I’m talkative and expressive."
    ],
    "Emotion regulation via music": [
        "I listen to music when I feel stressed or upset.",
        "Music helps me process difficult emotions.",
        "Intense or dark music can calm me.",
        "I feel recharged after my favorite songs.",
        "Music is a key part of my emotional life."
    ],
    "Sensation-seeking": [
        "I seek thrills and excitement.",
        "I enjoy fast, loud, or aggressive music.",
        "I like trying intense new experiences.",
        "I often push myself outside my comfort zone.",
        "I get bored when life is predictable."
    ]
}

# lets claculate likert scale 1-5, because we are psychologists :) :)
def likert_block(label, items):
    vals = []
    with st.container(border=True):
        st.markdown(f"**{label}**")
        for i, q in enumerate(items, start=1):
            vals.append(st.slider(q, 1, 5, 3, key=f"{label}_{i}"))
    scaled = [(v-1)/4 for v in vals]
    return float(np.mean(scaled))

cols = st.columns(3)
listener = {}
for i, trait in enumerate(TRAIT_NAMES):
    with cols[i % 3]:
        listener[trait] = likert_block(trait, QUESTIONS[trait])

listener_vec = np.array([listener[t] for t in TRAIT_NAMES])

st.subheader("2) Your trait profile")
st.dataframe(pd.DataFrame({"Trait": TRAIT_NAMES, "Score (0–1)": listener_vec}).set_index("Trait"))


st.subheader("3) Top subgenre matches")
g_names = list(genre_map.keys())
G = np.array([genre_map[g] for g in g_names], dtype=float)
sims = G @ listener_vec / (norm(G, axis=1) * norm(listener_vec))
top_idx = np.argsort(-sims)[:5]
top = [(g_names[i], float(sims[i])) for i in top_idx]
st.dataframe(pd.DataFrame(top, columns=["Subgenre","Similarity (0–1)"]))



# klapa....
st.subheader("4) Band examples")
def bands_for_subgenre(sub, k=6):
    sub_norm = sub.lower()
    bands_df = load_bands("demo_bands.csv")
    mask = bands_df["Genre_norm"].str.contains(sub_norm, regex=False, na=False)
    if mask.sum() < k and " " in sub_norm:
        for token in sub_norm.split():
            if len(token) > 3:
                mask |= bands_df["Genre_norm"].str.contains(token, regex=False, na=False)
    return bands_df.loc[mask, ["Band","Country","Genre"]].drop_duplicates().head(k)

for sub, score in top:
    st.markdown(f"### {sub} — similarity **{score:.3f}**")
    ex = bands_for_subgenre(sub)
    st.dataframe(ex.reset_index(drop=True))

#blablabla
st.caption("Exploratory recommender based on psychology correlations, not a clinical tool and solely for purpse of Crazy Data Science.")
