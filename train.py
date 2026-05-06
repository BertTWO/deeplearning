"""
train.py - Model training script using MobileNetV2 transfer learning.

Project structure:
    deeplearning/
    |-- tf_classroom_data/
    |   |-- train/
    |   |   |-- headphones/
    |   |   |-- mug/
    |   |   `-- calculator/
    |   |-- validation/
    |   |   |-- headphones/
    |   |   |-- mug/
    |   |   `-- calculator/
    |   `-- test/
    |       |-- headphones/
    |       |-- mug/
    |       `-- calculator/
    |-- train.py        <-- this file
    |-- camera.py
    |-- model.keras     (auto-generated after training)
    `-- class_names.txt (auto-generated after training)

Usage:
    python train.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DATA_DIR   = "./tf_classroom_data"
MODEL_PATH = "./model.keras"
CLASS_PATH = "./class_names.txt"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30
SEED       = 42


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_dataset(split):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="categorical",
    )


def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def optimize_pipeline(ds, shuffle=False):
    autotune = tf.data.AUTOTUNE
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.prefetch(autotune)


# ------------------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------------------

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

def build_model(num_classes):
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = layers.Input(shape=(*IMG_SIZE, 3))
    x       = augmentation(inputs)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

def get_callbacks():
    return [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    print("\nLoading datasets...")
    train_ds = load_dataset("train")
    val_ds   = load_dataset("validation")

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    with open(CLASS_PATH, "w") as f:
        f.write("\n".join(class_names))

    train_ds = optimize_pipeline(train_ds, shuffle=True)
    val_ds   = optimize_pipeline(val_ds)

    model, base = build_model(num_classes)

    # Phase 1: Train classification head only
    print("\nPhase 1: Training head (base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=get_callbacks())

    # Phase 2: Fine-tune top layers of the base model
    print("\nPhase 2: Fine-tuning top layers...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=get_callbacks())

    print("\nEvaluating on validation set...")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy : {acc * 100:.1f}%")
    print(f"Model saved to      : {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_PATH}")
    print("\nNext step: python camera.py")


if __name__ == "__main__":
    main()
