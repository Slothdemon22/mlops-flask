import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow

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
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5
LABEL_SMOOTHING = 0.1
FINE_TUNE_LAST_LAYERS = 50

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

# =========================
# 🔄 Prefetch
# =========================
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

# =========================
# ⚙️ Compile
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

# =========================
# 🛠️ Callbacks
# =========================
timestamp = int(time.time())
model_dir = f"models/run_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
checkpoint_path = os.path.join(model_dir, "best_model.keras")

# Custom MLflow callback to log metrics after each epoch
class MlflowEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

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
    ),
    MlflowEpochLogger()
]

# =========================
# 🚀 MLflow Setup
# =========================
mlflow.set_experiment("xception_fake_real")

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "dropout": DROPOUT,
        "optimizer": "Adam",
        "learning_rate_phase1": LEARNING_RATE_PHASE1,
        "learning_rate_phase2": LEARNING_RATE_PHASE2,
        "label_smoothing": LABEL_SMOOTHING,
        "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS
    })
    
    # =========================
    # 🚀 Train Phase 1
    # =========================
    model.fit(
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
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks
    )
    
    # =========================
    # 💾 Save final model locally
    # =========================
    final_model_path = os.path.join(model_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"✅ Model saved → {final_model_path}")
    
    # =========================
    # 📊 Log final metrics
    # =========================
    val_loss, val_acc = model.evaluate(val_ds)
    mlflow.log_metrics({
        "final_val_loss": float(val_loss),
        "final_val_accuracy": float(val_acc)
    })
    
    print(f"✅ MLflow run completed: {run.info.run_id}")
    print(f"🎉 MLflow run URL (your tracking server): {run.info.run_id}")

