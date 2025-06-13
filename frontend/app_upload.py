"""
Streamlit UI (upload-only) – Fruit & Veg Scanner
• Pengguna meng-upload foto, backend /predict men-return JSON hasil
• Sidebar menampilkan akurasi & confusion matrix (jika endpoint /model-metrics tersedia)
"""

import os, json, requests, io
import streamlit as st
import pandas as pd, altair as alt
try:
    import tomllib              # Py ≥3.11
except ModuleNotFoundError:
    import tomli as tomllib     # pip install tomli

# ─── Config ───────────────────────────────────────────────────────────────
with open("config.toml", "rb") as f:
    CFG = tomllib.load(f)

API_URL = os.getenv("API_URL", CFG["api"]["url"])
GREEN_T = CFG["ui"]["confidence_green"]
YELL_T  = CFG["ui"]["confidence_yellow"]

def color_badge(conf):
    if conf >= GREEN_T: return "🟢"
    if conf >= YELL_T:  return "🟡"
    return "🔴"

# ─── Sidebar: metrics (opsional) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_metrics():
    try:
        r = requests.get(API_URL.replace("/predict", "/model-metrics"), timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

with st.sidebar:
    st.title("📊 Evaluasi Model")
    metrics = fetch_metrics()
    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f} %")
        df = pd.DataFrame({
            "class"    : metrics["per_class"]["labels"],
            "precision": metrics["per_class"]["precision"],
            "recall"   : metrics["per_class"]["recall"],
            "f1"       : metrics["per_class"]["f1"],
        }).sort_values("recall", ascending=False)
        st.dataframe(df, height=250)
        cm = pd.DataFrame(metrics["confusion_matrix"],
                          index=metrics["per_class"]["labels"],
                          columns=metrics["per_class"]["labels"]).reset_index().melt(id_vars="index")
        st.altair_chart(
            alt.Chart(cm).mark_rect().encode(
                x="variable:N", y="index:N",
                color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"))
            ).properties(height=300),
            use_container_width=True
        )
    else:
        st.info("Model metrics belum tersedia.")

# ─── Main upload UI ────────────────────────────────────────────────────────
st.title("📥 Fruit-Veg Scanner – Upload Foto")

uploaded = st.file_uploader(
    "Unggah gambar buah / sayuran (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

result_box = st.empty()
if uploaded and st.button("Predict"):
    st.image(uploaded, caption="Gambar di-upload", use_column_width=True)

    # kirim ke API
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    with st.spinner("Memproses…"):
        try:
            resp = requests.post(API_URL, files=files, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Gagal memanggil API: {e}")
            st.stop()

    badge   = color_badge(data["confidence"])
    confpct = data["confidence"]*100
    result_box.markdown(
        f"## {badge} **{data['name'].title()}** – Rp{data['price']:,}"
        f"<br/>Confidence: **{confpct:.1f}%**",
        unsafe_allow_html=True
    )

    # Kandidat alternatif
    if "candidates" in data:
        st.warning("Model kurang yakin – pilih kandidat manual:")
        for cand in data["candidates"]:
            if st.button(f"{cand['name'].title()} ({cand['conf']*100:.1f} %)"):
                result_box.markdown(
                    f"## 🟢 **{cand['name'].title()}** – Rp{cand['price']:,} (dipilih manual)",
                    unsafe_allow_html=True
                )
                break
