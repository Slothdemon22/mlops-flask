import tensorflow as tf
from tensorflow import keras
import json

# =========================
# 📁 Paths
# =========================
test_dir = "data_processed/test"
model_path = "best_model.keras"

# =========================
# 📦 Load Dataset
# =========================
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =========================
# 🧩 Load Model
# =========================
model = keras.models.load_model(model_path)

# =========================
# 🔥 Evaluate
# =========================
loss, acc = model.evaluate(test_ds)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

# =========================
# 💾 Save metrics in DVC format
# =========================
metrics = {
    "loss": float(loss),
    "accuracy": float(acc)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved → metrics.json")
