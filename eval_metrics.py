# eval_metrics.py
import tensorflow as tf, pathlib, json, numpy as np, sklearn.metrics as skm, matplotlib.pyplot as plt, seaborn as sns

MODEL = "model/fruitveg_finetune.keras"          # FP32 atau .tflite (pakai tf.lite.Interpreter)
DATA  = pathlib.Path("data/dataset/test")        # set test
IMG   = (224, 224)

# ---------- load dataset ----------
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA, image_size=IMG, batch_size=32, shuffle=False)
class_names = test_ds.class_names

# ---------- load model ----------
model = tf.keras.models.load_model(MODEL)

# ---------- inference ----------
y_true, y_pred = [], []
for xb, yb in test_ds:
    y_true.extend(yb.numpy())
    pred = tf.argmax(model(xb, training=False), 1).numpy()
    y_pred.extend(pred)

# ---------- metrics ----------
accuracy = float(skm.accuracy_score(y_true, y_pred))
precision = skm.precision_score(y_true, y_pred, average=None).tolist()
recall    = skm.recall_score(y_true, y_pred, average=None).tolist()
f1        = skm.f1_score(y_true, y_pred, average=None).tolist()
cm        = skm.confusion_matrix(y_true, y_pred).tolist()   # for heatmap

json.dump(
    {
        "accuracy": accuracy,
        "per_class": {
            "labels": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "confusion_matrix": cm
    },
    open("model/metrics.json", "w"), indent=2)
print("✓ metrics.json saved")
