import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, time
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import numpy as np
import shutil

# =========================
# 📁 Paths
# =========================
train_dir = "data_processed/train"
val_dir   = "data_processed/val"

# =========================
# ⚙️ Parameters
# =========================
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 10
SEED = 42
DROPOUT = 0.3
LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5
LABEL_SMOOTHING = 0.1
FINE_TUNE_LAST_LAYERS = 50

# =========================
# 🔐 DagsHub Credentials
# =========================
DAGSHUB_USERNAME = "Slothdemon22"
DAGSHUB_TOKEN = "a50bc6e3b74b10f5c959b12bce5de638474e7b48"
REPO_NAME = "mlops-project"
EXPERIMENT_NAME = "xception_fake_real"

os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# 🔄 Data Augmentation
# =========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# =========================
# 📦 Load datasets
# =========================
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# =========================
# 🧩 Build MODEL
# =========================
base_model = keras.applications.Xception(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.xception.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

# =========================
# 🛠️ Callbacks & Checkpoints
# =========================
timestamp = int(time.time())
model_dir = f"models/run_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
checkpoint_path = os.path.join(model_dir, "best_model.keras")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=3,
        verbose=1
    )
]

# =========================
# 🚀 Training Phase 1
# =========================
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks
)

# =========================
# 🔓 Fine-tuning Phase 2
# =========================
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks
)

# =========================
# 💾 Save final model
# =========================
final_model_path = os.path.join(model_dir, "final_model.keras")
model.save(final_model_path)
print(f"✅ Model saved → {final_model_path}")

# =========================
# 📊 MLflow Logging to DagsHub
# =========================
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Log hyperparameters
    mlflow.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "dropout": DROPOUT,
        "optimizer": "Adam",
        "learning_rate_phase1": LR_PHASE1,
        "learning_rate_phase2": LR_PHASE2,
        "label_smoothing": LABEL_SMOOTHING,
        "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS
    })

    # Log metrics from both phases
    for epoch, acc in enumerate(history1.history["accuracy"], 1):
        mlflow.log_metric("train_accuracy_phase1", acc, step=epoch)
    for epoch, val_acc in enumerate(history1.history["val_accuracy"], 1):
        mlflow.log_metric("val_accuracy_phase1", val_acc, step=epoch)
    for epoch, acc in enumerate(history2.history["accuracy"], 1):
        mlflow.log_metric("train_accuracy_phase2", acc, step=epoch)
    for epoch, val_acc in enumerate(history2.history["val_accuracy"], 1):
        mlflow.log_metric("val_accuracy_phase2", val_acc, step=epoch)

    # Log model with signature
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32) * 255
    pred = model.predict(dummy_input, verbose=0)
    signature = infer_signature(dummy_input, pred)

    try:
        mlflow.keras.log_model(model, artifact_path="final_model", signature=signature)
    except Exception:
        temp_dir = f"temp_model_artifact"
        mlflow.keras.save_model(model, temp_dir, signature=signature)
        mlflow.log_artifacts(temp_dir, artifact_path="final_model")
        shutil.rmtree(temp_dir)

    # Log original model file
    mlflow.log_artifact(final_model_path, artifact_path="original_model")

print(f"🎉 SUCCESS! Model + training run logged to DagsHub!")
print(f"📊 Run URL: https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow/#/experiments/3/runs/{run_id}")
