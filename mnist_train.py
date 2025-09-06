"""
mnist_train.py
Simple MNIST handwritten digit recognition training script using Keras (TensorFlow).
Saves model, metrics, and plots to outputs/.
Designed to run on CPU/GPU. Use small epochs (5-10) for quick runs on CPU.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Config ---
RANDOM_STATE = 42
EPOCHS = 8           # Reduce to 5 if your CPU is slow; increase if you want better accuracy
BATCH_SIZE = 128
OUTPUT_DIR = Path("outputs")
SAMPLE_COUNT = 10    # how many test images to show in the sample_predictions grid
# ----------------

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_environment_info():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible to TF:", len(gpus))
    if gpus:
        print("GPU devices:", gpus)

def load_and_preprocess():
    # Load MNIST from Keras datasets (downloads automatically first run)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("Raw shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Normalize to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Keep shape (N, 28, 28) for Flatten layer
    # One-hot encode labels
    y_train_ohe = to_categorical(y_train, num_classes=10)
    y_test_ohe = to_categorical(y_test, num_classes=10)

    print("After preprocessing shapes:", x_train.shape, y_train_ohe.shape, x_test.shape, y_test_ohe.shape)
    return (x_train, y_train_ohe), (x_test, y_test_ohe), (y_test)  # return raw y_test for comparisons

def build_model():
    # Simple fully-connected (dense) model: Flatten -> Dense -> Output
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def plot_and_save_history(history):
    # Plot accuracy and loss
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="train_acc")
    plt.plot(epochs_range, val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="train_loss")
    plt.plot(epochs_range, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    path = OUTPUT_DIR / "training_history.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training history saved to: {path}")

def save_metrics(test_loss, test_acc):
    mpath = OUTPUT_DIR / "metrics.txt"
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_accuracy: {test_acc:.6f}\n")
    print(f"Metrics saved to: {mpath}")

def save_sample_predictions(model, x_test, y_test_raw):
    # Predict first SAMPLE_COUNT images and save a grid
    n = min(SAMPLE_COUNT, x_test.shape[0])
    preds = model.predict(x_test[:n])
    pred_labels = np.argmax(preds, axis=1)
    true_labels = y_test_raw[:n]

    # Plot a grid (2 columns)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_test[i], cmap="gray", vmin=0, vmax=1)
        color = "green" if pred_labels[i] == true_labels[i] else "red"
        plt.title(f"P:{pred_labels[i]} / T:{true_labels[i]}", color=color)
        plt.axis("off")
    out = OUTPUT_DIR / "sample_predictions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Sample predictions saved to: {out}")

def main():
    ensure_dirs()
    print_environment_info()

    (x_train, y_train), (x_test, y_test_ohe), y_test_raw = load_and_preprocess()
    model = build_model()

    # Callbacks: save best model by validation accuracy
    checkpoint_path = OUTPUT_DIR / "best_model.h5"
    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test_ohe),
        callbacks=callbacks,
        verbose=2
    )

    plot_and_save_history(history)

    # Evaluate on test
    test_loss, test_acc = model.evaluate(x_test, y_test_ohe, verbose=0)
    print(f"\nTest loss: {test_loss:.6f}  |  Test accuracy: {test_acc:.6f}")

    save_metrics(test_loss, test_acc)

    # Save final model (the best model was saved by ModelCheckpoint as best_model.h5)
    final_model_path = OUTPUT_DIR / "final_model.h5"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save sample predictions
    save_sample_predictions(model, x_test, y_test_raw)

    print("\nAll done. Open the outputs/ folder to view results.")

if __name__ == "__main__":
    main()
