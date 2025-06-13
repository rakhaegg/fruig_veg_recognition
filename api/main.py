"""
FastAPI service “/predict” untuk klasifikasi buah & sayur + lookup harga
– Model  : model/fruitveg_int8.tflite  (INT-8, 224×224)
– Harga  : data/price_list.csv         (kolom: sku,price)
– Dataset: data/dataset/train/<kelas>/…  ← hanya dipakai ambil nama label
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
from pathlib import Path
import logging, time

# ────────────────────────────────────────────────────────────────────────────
# Konstanta & jalur
MODEL_PATH  = Path("model/fruitveg_int8.tflite")
PRICE_PATH  = Path("data/price_list.csv")
LABEL_ROOT  = Path("data/dataset/train")      # ambil nama folder kelas
IMG_SIZE    = (224, 224)
CONF_THRESH = 0.60                            # confidence minimum
# ────────────────────────────────────────────────────────────────────────────

# ─── Startup: load label, harga, model ─────────────────────────────────────
LABELS = [d.name for d in sorted(LABEL_ROOT.iterdir())]

price_df  = pd.read_csv(PRICE_PATH)
PRICE_MAP = dict(zip(price_df["sku"], price_df["price"]))

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
IN_IDX  = interpreter.get_input_details()[0]["index"]
OUT_IDX = interpreter.get_output_details()[0]["index"]

# ─── FastAPI instance ──────────────────────────────────────────────────────
app = FastAPI(title="Fruit & Veg Classifier API", version="1.0")

# CORS (boleh di‐trim ke domain internal POS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ─── Middleware log latensi ────────────────────────────────────────────────
logger = logging.getLogger("uvicorn.access")
@app.middleware("http")
async def log_time(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %.0f ms %s",
        request.url.path,
        (time.perf_counter() - t0) * 1000,
        response.status_code,
    )
    return response

# ─── Util: preprocess gambar → batch uint8 ────────────────────────────────
def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.uint8)
    return arr[np.newaxis, ...]         # (1,224,224,3)

# ─── Endpoint utama ───────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "File harus JPEG/PNG")

    batch = preprocess(await file.read())

    interpreter.set_tensor(IN_IDX, batch)
    interpreter.invoke()
    probs = interpreter.get_tensor(OUT_IDX)[0] / 255.0      # uint8 → [0,1]

    top = int(probs.argmax())
    conf = float(probs[top])
    name = LABELS[top]
    price = PRICE_MAP.get(name)

    result = {"name": name, "confidence": conf, "price": price}

    if conf < CONF_THRESH:            # kirim 3 kandidat jika ragu
        top3 = probs.argsort()[-3:][::-1]
        result["candidates"] = [
            {"name": LABELS[i], "conf": float(probs[i]), "price": PRICE_MAP.get(LABELS[i])}
            for i in top3
        ]
    return result
