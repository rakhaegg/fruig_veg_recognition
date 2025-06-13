"""
convert_tflite.py
▪ Memuat model/fruitveg.keras
▪ Full-integer quantization memakai 100 contoh gambar
▪ Menyimpan fruitveg_int8.tflite
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from random import sample
from tqdm import tqdm
from PIL import Image

MODEL_PATH = Path("model/fruitveg.keras")
DATA_DIR   = Path("data/dataset/train")      # pakai set train utk representative
TFLITE_OUT = Path("model/fruitveg_int8.tflite")
IMG_SIZE   = (224, 224)
REP_SAMPLES = 100                            # 100 gambar cukup utk kalibrasi

# ---------- 1. Representative dataset generator ----------
def rep_data_gen():
    # ambil 100 file acak dari seluruh kelas
    all_imgs = list(DATA_DIR.rglob("*.*"))
    for img_path in tqdm(sample(all_imgs, REP_SAMPLES), desc="Collect rep data"):
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
        yield [arr]

# ---------- 2. Buat converter ----------
converter = tf.lite.TFLiteConverter.from_keras_model(
    tf.keras.models.load_model(MODEL_PATH)
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]          # aktifkan quant
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

print("⏳  Mengonversi ke TFLite INT8...")
tflite_model = converter.convert()
TFLITE_OUT.write_bytes(tflite_model)
print(f"✅  Selesai. File: {TFLITE_OUT}  |  Size: {TFLITE_OUT.stat().st_size/1e6:.2f} MB")
