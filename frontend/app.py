"""
Streamlit UI – Fruit & Veg Scanner
• Webcam snapshot memakai st.camera_input() (tidak tergantung OpenCV)
• Kirim gambar ke FastAPI /predict → tampilkan nama, harga, confidence
"""

from pathlib import Path
import time
import requests
import streamlit as st
import pandas as pd, altair as alt, requests, json

# ─── Baca config.toml (tomllib ≥ Py3.11, tomli utk 3.10↓) ──────────────────
try:
    import tomllib           # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli  (hanya bila Py < 3.11)

with open("config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

API_URL   = CONFIG["api"]["url"]
GREEN_T   = CONFIG["ui"]["confidence_green"]
YELL_T    = CONFIG["ui"]["confidence_yellow"]

# ─── Streamlit settings ────────────────────────────────────────────────────
st.set_page_config(page_title="Fruit-Veg Scanner", page_icon="🍎", layout="centered")
st.title("🍏 Fruit & Veg Scanner")

# Sidebar untuk menyesuaikan threshold warna
with st.sidebar:
    st.header("Pengaturan")
    GREEN_T = st.slider("Threshold hijau ≥", 0.7, 1.0, float(GREEN_T), 0.01)
    YELL_T  = st.slider("Threshold kuning ≥", 0.5, GREEN_T, float(YELL_T), 0.01)
    st.caption("Merah jika confidence di bawah kuning")

# ─── Helper util ───────────────────────────────────────────────────────────
def color_badge(conf: float) -> str:
    """Return colored emoji bullet."""
    if conf >= GREEN_T:
        return "🟢"
    if conf >= YELL_T:
        return "🟡"
    return "🔴"

def query_api(img_bytes: bytes) -> dict:
    """POST image bytes to backend API and return JSON."""
    resp = requests.post(API_URL,
                         files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                         timeout=10)
    resp.raise_for_status()
    return resp.json()

# ─── UI area ───────────────────────────────────────────────────────────────
st.subheader("1 • Ambil foto")
img_file = st.camera_input("Klik *Take Photo* lalu *Capture*")

result_box = st.empty()   # tempat menampilkan output

if img_file and st.button("⏩ Predict"):
    # 1. tampilkan thumbnail
    st.image(img_file, caption="Snapshot", use_column_width=True)

    # 2. panggil API
    t0 = time.perf_counter()
    data = query_api(img_file.getvalue())
    latency_ms = (time.perf_counter() - t0) * 1000

    # 3. render hasil utama
    badge = color_badge(data["confidence"])
    conf_pct = data["confidence"] * 100
    result_box.markdown(
        f"## {badge} **{data['name'].title()}**  –  Rp{data['price']:,}"
        f"<br/>Confidence: **{conf_pct:.1f}%** | Latency: {latency_ms:.0f} ms",
        unsafe_allow_html=True,
    )

    # 4. jika ragu, tampilkan kandidat tombol
    if "candidates" in data:
        st.warning("Model kurang yakin — pilih salah satu kandidat di bawah:")
        cols = st.columns(len(data["candidates"]))
        for col, cand in zip(cols, data["candidates"]):
            with col:
                if st.button(
                    f"{cand['name'].title()} ({cand['conf']*100:.1f} %)"
                ):
                    result_box.markdown(
                        f"## 🟢 **{cand['name'].title()}**  –  Rp{cand['price']:,} "
                        "(dipilih manual)",
                        unsafe_allow_html=True,
                    )

@st.cache_data(show_spinner=False)
def get_metrics():
    try:
        r = requests.get(API_URL.replace("/predict", "/model-metrics"), timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

metrics = get_metrics()

with st.sidebar:
    st.header("📊 Evaluasi Model")
    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f} %")

        # Tabel precision / recall
        df = pd.DataFrame({
            "class": metrics["per_class"]["labels"],
            "precision": metrics["per_class"]["precision"],
            "recall":    metrics["per_class"]["recall"],
            "f1":        metrics["per_class"]["f1"],
        }).sort_values("recall", ascending=False)
        st.dataframe(df, height=240)

        # Confusion matrix heatmap (Altair)
        cm = pd.DataFrame(metrics["confusion_matrix"],
                          index=metrics["per_class"]["labels"],
                          columns=metrics["per_class"]["labels"]).reset_index().melt(id_vars="index")
        heat = alt.Chart(cm).mark_rect().encode(
            x=alt.X('variable:N', title='Predicted'),
            y=alt.Y('index:N', title='True'),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='blues'))
        ).properties(height=300)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Model metrics belum tersedia.")