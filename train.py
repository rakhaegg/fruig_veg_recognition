# train_finetune.py
# Fine-tune EfficientNetB0 pada dataset buah-sayur + Confusion Matrix TensorBoard
import tensorflow as tf
import pathlib, itertools, io, math
import matplotlib.pyplot as plt
import sklearn.metrics as skm

# ─── Parameter utama ───────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH       = 32
WARM_EPOCHS = 3            # tahap backbone frozen
FT_EPOCHS   = 9            # tahap fine-tune (total = 12)
UNFREEZE_LAYERS = 80       # layer terakhir yang dibuka
DATA_DIR    = pathlib.Path("data/dataset")
LOG_DIR     = "logs_ft"
SEED        = 42

# ─── Callback confusion matrix ─────────────────────────────────────────────
class ConfMatrixCB(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, class_names):
        super().__init__()
        self.val_ds, self.names = val_ds, class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x, y in self.val_ds:
            y_true.extend(y.numpy())
            y_pred.extend(tf.argmax(self.model(x, training=False), 1).numpy())
        cm = skm.confusion_matrix(y_true, y_pred, labels=range(len(self.names)))
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.names))); ax.set_xticklabels(self.names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(self.names))); ax.set_yticklabels(self.names, fontsize=6)
        thresh = cm.max() / 2
        for i, j in itertools.product(*map(range, cm.shape)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        img = tf.image.decode_png(buf.getvalue(), channels=4)[None]
        tf.summary.image("Confusion_Matrix", img, step=epoch)

# ─── Data pipeline ─────────────────────────────────────────────────────────
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH, seed=SEED)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "val", image_size=IMG_SIZE, batch_size=BATCH, seed=SEED)
class_names = train_ds.class_names

aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])
train_ds = train_ds.map(lambda x, y: (aug(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ─── Model: EfficientNetB0 + neck kecil ───────────────────────────────────
base = tf.keras.applications.EfficientNetB0(include_top=False,
                                            weights="imagenet",
                                            input_shape=(*IMG_SIZE, 3),
                                            pooling="avg")
base.trainable = False   # tahap warm-up

inputs = tf.keras.Input((*IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.Reshape((1, 1, -1))(x)
x = tf.keras.layers.Conv2D(1280, 1, activation="relu")(x)  # neck
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="efnetb0_finetune")

# ─── Callbacks umum ────────────────────────────────────────────────────────
cm_cb = ConfMatrixCB(val_ds, class_names)
tb_cb = tf.keras.callbacks.TensorBoard(LOG_DIR)

# ─── Tahap 1: warm-up (backbone beku) ──────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARM_EPOCHS,
    callbacks=[tb_cb, cm_cb],
)

# ─── Tahap 2: buka sebagian layer & fine-tune ──────────────────────────────
for layer in base.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# scheduler cosine decay
def cosine_lr(epoch):
    e = epoch  # 0..FT_EPOCHS-1
    return 5e-4 * 0.5 * (1 + math.cos(e / (FT_EPOCHS - 1) * math.pi))

sched_cb = tf.keras.callbacks.LearningRateScheduler(cosine_lr)

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=WARM_EPOCHS,
    epochs=WARM_EPOCHS + FT_EPOCHS,
    callbacks=[tb_cb, cm_cb, sched_cb],
)

# ─── Simpan model ──────────────────────────────────────────────────────────
pathlib.Path("model").mkdir(exist_ok=True)
model.save("model/fruitveg_finetune.keras")
print("✓ Training selesai ➜ model/fruitveg_finetune.keras")
