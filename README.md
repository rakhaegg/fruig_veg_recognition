# 🍏 Sistem Pengenalan Buah & Sayur

Panduan lengkap—dari instalasi, pelatihan ulang, hingga deployment Docker—dalam Bahasa Indonesia.

---

## 1 · Jalankan Demo Cepat (Docker)

```bash
git clone https://github.com/your-org/fruit-veg-demo.git
cd fruit-veg-demo
docker compose up -d --build   # build + jalankan API & UI
```

* **API** : [http://localhost:8000/docs](http://localhost:8000/docs)
* **UI**  : [http://localhost:8501](http://localhost:8501)

Unggah foto (JPEG/PNG) dan lihat nama, harga (Rp), confidence, serta confusion‑matrix.

---

## 2 · Menjalankan Tanpa Docker

### a) Buat venv & install dependensi

```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" tensorflow-cpu==2.19 \
            pillow python-multipart streamlit altair requests tomli
```

### b) Jalankan

```bash
uvicorn api.main:app --port 8000          # Terminal A
streamlit run frontend/app_upload.py      # Terminal B
```

---

## 3 · Struktur Folder

```
api/            → FastAPI backend
frontend/       → UI Streamlit (upload)
model/…         → *.tflite & metrics.json
data/           → price_list.csv + label map
train_finetune.py  convert_tflite.py  eval_metrics.py
Dockerfile.api  Dockerfile.ui  docker-compose.yml  config.toml
```

---

## 4 · Melatih Ulang Model

```bash
python train_finetune.py      # hasil .keras
python convert_tflite.py      # hasil .tflite INT‑8
python eval_metrics.py        # metrics.json
```

Salin `model/fruitveg_int8.tflite` baru ke `api/model/…` → restart API.

---

## 5 · Memperbarui Harga

```
vim data/price_list.csv       # ubah harga per kg
docker compose restart api
```

---

## 6 · Endpoint Penting

| Metode | Path           | Fungsi                                  |
| ------ | -------------- | --------------------------------------- |
| POST   | /predict       | Gambar → JSON {name, price, confidence} |
| GET    | /model-metrics | Akurasi global + confusion‑matrix       |
| GET    | /health        | Cek status service                      |

---

## 7 · Ukuran Image Kecil (API)

```dockerfile
FROM python:3.10-slim
RUN pip install --no-cache-dir fastapi uvicorn tflite-runtime pillow python-multipart
# image ≈ 180 MB
```

---

## 8 · Sprint Roadmap

1. Data Pipeline → baseline
2. Fine‑tune EfficientNet + INT‑8
3. API FastAPI
4. UI Streamlit
5. Docker pilot & log
6. Active learning & OTA update


