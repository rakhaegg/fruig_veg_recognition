"""
train.py — Latih model klasifikasi 36 jenis buah & sayur
• Dataset folder: data/dataset/{train,val,test}/{class}/img.jpg
• Output:
    model/best.keras      – checkpoint val_accuracy terbaik
    model/fruitveg.keras  – model akhir (selesai semua epoch)
"""

from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/dataset")     
IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 3                         
SEED     = 42

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="int",
    seed=SEED,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="int",
    seed=SEED,
)
class_names = train_ds.class_names          

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(0.15),
    ],
    name="augment",
)
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
    pooling="avg",
    name="efficientnetb0",
)
base.trainable = False    

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image")
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="fruitveg_clf")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

Path("model").mkdir(exist_ok=True)

ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "model/best.keras",           
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1,
)
early_cb = tf.keras.callbacks.EarlyStopping(
    patience=2,
    monitor="val_accuracy",
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[ckpt_cb, early_cb],
)

model.save("model/fruitveg.keras")   
print("Training selesai. Model disimpan di model/fruitveg.keras")
