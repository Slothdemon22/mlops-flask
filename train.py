import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import time
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
EPOCHS = 20
SEED = 42

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
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# =========================
# ⚙️ Compile
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# =========================
# 🛠️ Callbacks
# =========================
timestamp = int(time.time())
model_dir = f"models/run_{timestamp}"
os.makedirs(model_dir, exist_ok=True)

# Only save the best model for this run, not every epoch globally
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
# 🚀 Train Phase 1
# =========================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# 🔓 Fine-tuning Phase 2
# =========================
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# =========================
# 💾 Save final model
# =========================
final_model_path = os.path.join(model_dir, "final_model.keras")
model.save(final_model_path)
print(f"Model saved → {final_model_path}")

# =========================
# 📊 MLflow logging (params, metrics, hyperparams only)
# =========================
mlflow.set_experiment("xception_fake_real")

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS,
        "epochs_phase2": 10,
        "dropout": 0.3,
        "optimizer": "Adam",
        "learning_rate_phase1": 1e-4,
        "learning_rate_phase2": 1e-5,
        "label_smoothing": 0.1,
        "fine_tune_last_layers": 50
    })
    
    # Log metrics from last evaluation
    loss, acc = model.evaluate(val_ds)
    mlflow.log_metrics({
        "val_loss": float(loss),
        "val_accuracy": float(acc)
    })
    
    print(f"✅ MLflow run logged: {run.info.run_id}")
