# app.py
# Streamlit CBF Recommender — Wisata Surabaya
# Author: Labiba Adinda Zahwana
# Email : labibaadinda11@gmail.com

import os
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---- Import otomatis Sastrawi ----
def _safe_import_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        return StemmerFactory, StopWordRemoverFactory
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Sastrawi"], check=False)
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        return StemmerFactory, StopWordRemoverFactory

StemmerFactory, StopWordRemoverFactory = _safe_import_sastrawi()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

# --------------------------
# CONFIGURASI STREAMLIT
# --------------------------
st.set_page_config(page_title="Rekomendasi Wisata Surabaya", page_icon="🧭", layout="wide")

# --------------------------
# 1. LOAD DATA (LOCAL)
# --------------------------
@st.cache_data(show_spinner=False)
def load_local_data():
    try:
        pkg = pd.read_csv("package_tourism.csv")
        rating = pd.read_csv("tourism_rating.csv")
        dest = pd.read_csv("tourism_with_id.csv")
        user = pd.read_csv("user.csv")
        return pkg, rating, dest, user
    except FileNotFoundError as e:
        st.error("❌ File tidak ditemukan. Pastikan semua dataset (.csv) berada di folder yang sama dengan app.py")
        st.stop()

# --------------------------
# 2. PREPROCESSING FUNCTION
# --------------------------
def build_preprocess_funcs():
    stemmer = StemmerFactory().create_stemmer()
    stop_rm = StopWordRemoverFactory().create_stop_word_remover()
    def preprocessing(text: str) -> str:
        if pd.isna(text):
            return ""
        t = text.lower()
        t = stemmer.stem(t)
        t = stop_rm.remove(t)
        return t
    return preprocessing

@st.cache_data(show_spinner=False)
def prepare_content_data(dest_df: pd.DataFrame):
    cols_drop = ['Time_Minutes','Coordinate','Lat','Long','Unnamed: 11','Unnamed: 12']
    for c in cols_drop:
        if c in dest_df.columns:
            dest_df = dest_df.drop(columns=c)
    place_sby = dest_df[dest_df['City'] == 'Surabaya'].copy()

    preprocessing = build_preprocess_funcs()
    place_sby['Description'] = place_sby['Description'].astype(str).apply(preprocessing)
    place_sby['Category'] = place_sby['Category'].astype(str).apply(preprocessing)
    place_sby['Tags'] = (place_sby['Description'] + " " + place_sby['Category']).str.strip()

    df_cbf = place_sby[['Place_Id','Place_Name','Tags']].reset_index(drop=True)
    return place_sby.reset_index(drop=True), df_cbf

@st.cache_data(show_spinner=False)
def build_tfidf_and_similarity(tags_series: pd.Series):
    tv = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf = tv.fit_transform(tags_series)
    sim = cosine_similarity(tfidf, tfidf)
    return tv, tfidf, sim

@st.cache_data(show_spinner=False)
def average_rating_per_place(rating_df: pd.DataFrame, place_ids: pd.Series):
    r = rating_df[rating_df["Place_Id"].isin(place_ids)].copy()
    avg = r.groupby("Place_Id")["Place_Ratings"].mean().reset_index()
    avg.rename(columns={"Place_Ratings":"Avg_Rating"}, inplace=True)
    avg = pd.merge(place_ids.to_frame(), avg, on="Place_Id", how="left")
    avg["Avg_Rating"] = avg["Avg_Rating"].fillna(0.0)
    return avg

# --------------------------
# 3. RECOMMENDER FUNCTIONS
# --------------------------
def recommend_by_place(place_name: str, sim: np.ndarray, df_cbf: pd.DataFrame, topk: int = 10):
    idx_match = df_cbf.index[df_cbf['Place_Name'] == place_name]
    if len(idx_match) == 0:
        return pd.DataFrame(columns=["Place_Id", "Place_Name", "Cosine_Similarity"])
    idx = idx_match[0]
    scores = sim[idx].astype(float).copy()
    scores[idx] = -1e9
    order = np.argsort(scores)[::-1][:topk]
    out = df_cbf.loc[order, ['Place_Id', 'Place_Name']].copy()
    out['Cosine_Similarity'] = scores[order]
    return out.reset_index(drop=True)

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels / discounts))

def idcg_at_k(rels, k):
    rels_sorted = np.sort(rels)[::-1]
    return dcg_at_k(rels_sorted, k)

def ndcg_from_rels(rels_in_pred_order, k):
    dcg = dcg_at_k(rels_in_pred_order, k)
    idcg = idcg_at_k(rels_in_pred_order, k)
    return (dcg / idcg if idcg > 0 else 0.0), dcg, idcg

@st.cache_data(show_spinner=False)
def evaluate_itemwise_ndcg(sim: np.ndarray, avg_df: pd.DataFrame, k_list=(5,10)):
    graded_rel = np.clip((avg_df["Avg_Rating"].values - 1.0) / 4.0, 0.0, 1.0)
    binary_rel = (avg_df["Avg_Rating"].values >= 3.0).astype(float)
    n_items = len(avg_df)
    rows = []
    for i in range(n_items):
        scores = sim[i].astype(float).copy()
        scores[i] = -1e9
        order = np.argsort(scores)[::-1]
        rels_g = graded_rel[order]
        rels_b = binary_rel[order]
        for K in k_list:
            ndcg_g, dcg_g, idcg_g = ndcg_from_rels(rels_g, K)
            ndcg_b, dcg_b, idcg_b = ndcg_from_rels(rels_b, K)
            rows.append({
                "Item_Index": i, "K": K,
                "DCG_graded": dcg_g, "IDCG_graded": idcg_g, "NDCG_graded": ndcg_g,
                "DCG_binary": dcg_b, "IDCG_binary": idcg_b, "NDCG_binary": ndcg_b,
            })
    df_detail = pd.DataFrame(rows)
    summary = (
        df_detail.groupby("K")[["NDCG_graded","NDCG_binary","DCG_graded","IDCG_graded","DCG_binary","IDCG_binary"]]
        .mean().reset_index()
    )
    return summary, df_detail

# --------------------------
# 4. STREAMLIT UI
# --------------------------
st.title("🧭 Content-Based Recommender — Wisata Surabaya")
st.caption("Metode: TF-IDF + Cosine Similarity | Evaluasi: NDCG@K")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    TOPK = st.slider("Top-N rekomendasi", 5, 20, 10, 1)
    show_scores = st.toggle("Tampilkan skor cosine", value=True)
    show_avg = st.toggle("Tampilkan Avg_Rating", value=True)
    st.markdown("---")
    use_binary = st.toggle("Tampilkan NDCG biner (threshold ≥ 3)", value=True)
    K_list = st.multiselect("Cutoff K", [5,10,20], default=[5,10])

# Load Data
with st.spinner("Memuat data lokal..."):
    pkg, rating_df, dest_df, user_df = load_local_data()
    place_sby, df_cbf = prepare_content_data(dest_df)
    tv, tfidf, sim = build_tfidf_and_similarity(df_cbf["Tags"])
    avg_df = average_rating_per_place(rating_df, df_cbf["Place_Id"])

tab_rec, tab_eval = st.tabs(["🔎 Rekomendasi", "📈 Evaluasi"])

with tab_rec:
    st.subheader("Cari rekomendasi berdasarkan tempat")
    place_selected = st.selectbox("Pilih tempat:", options=df_cbf["Place_Name"].tolist())
    if place_selected:
        rec = recommend_by_place(place_selected, sim, df_cbf, topk=TOPK)
        rec = rec.merge(avg_df, on="Place_Id", how="left")
        show_cols = ["Place_Name"]
        if show_scores: show_cols.append("Cosine_Similarity")
        st.dataframe(rec[["Place_Id"]+show_cols], use_container_width=True)
        csv_rec = rec.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Rekomendasi", csv_rec, file_name="recommendations.csv", mime="text/csv")

with tab_eval:
    st.subheader("Evaluasi NDCG (item-wise)")
    if len(K_list) == 0:
        st.info("Pilih minimal satu nilai K di sidebar.")
    else:
        with st.spinner("Menghitung NDCG..."):
            summary, detail = evaluate_itemwise_ndcg(sim, avg_df, tuple(K_list))
        cols_to_show = ["K","NDCG_graded"]
        if use_binary: cols_to_show.append("NDCG_binary")
        st.dataframe(summary[cols_to_show], use_container_width=True)
        st.download_button(
            "⬇️ Unduh Hasil Lengkap",
            detail.to_csv(index=False).encode("utf-8"),
            file_name="ndcg_detail.csv",
            mime="text/csv"
        )

st.markdown("---")

st.caption("Dataset: [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data)")
st.caption("Source Code: [Github](https://github.com/labibaadinda/surabaya-tourism-destination-recommendation-system)")

