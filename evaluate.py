import tensorflow as tf
from tensorflow import keras
from pymongo import MongoClient
from datetime import datetime, timezone
import json

# =========================
# 📁 Paths & Mongo Config
# =========================
test_dir = "data_processed/test"
model_path = "best_model.keras"

# MongoDB (same creds as app.py, different collection)
MONGO_URI = "mongodb://admin:secret123@localhost:27017/predictions_db?authSource=admin"
client = MongoClient(MONGO_URI)
db = client["predictions_db"]
eval_collection = db["evaluations"]

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

# =========================
# 🗄️ Store in MongoDB
# =========================
try:
    eval_doc = {
        "collection": "evaluations",
        "model_path": model_path,
        "test_dir": test_dir,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc)
    }
    eval_collection.insert_one(eval_doc)
    print("✅ Evaluation metrics stored in MongoDB (evaluations collection)")
except Exception as e:
    print(f"❌ Failed to store metrics in MongoDB: {e}")
