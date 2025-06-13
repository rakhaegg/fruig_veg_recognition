from pathlib import Path
import tensorflow as tf, numpy as np, sys
from PIL import Image

IMG_PATH = sys.argv[1]
LABELS   = [d.name for d in sorted(Path("data/dataset/train").iterdir())]

# Load model
interpreter = tf.lite.Interpreter(model_path="model/fruitveg_int8.tflite")
interpreter.allocate_tensors()
input_idx  = interpreter.get_input_details()[0]["index"]
output_idx = interpreter.get_output_details()[0]["index"]

# Pre-process
img = Image.open(IMG_PATH).convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(img, dtype=np.uint8), 0)          # uint8 sesuai converter
interpreter.set_tensor(input_idx, x)
interpreter.invoke()
prob = interpreter.get_tensor(output_idx)[0]
print("Prediksi:", LABELS[int(prob.argmax())], " |  confident:", prob.max())
