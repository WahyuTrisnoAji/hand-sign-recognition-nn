"""
Training Model JST Backpropagation

Modul ini menjalankan seluruh pipeline training dari awal hingga model
tersimpan. Prosesnya meliputi pemuatan dataset, preprocessing data,
training jaringan syaraf tiruan, evaluasi hasil, dan penyimpanan model
beserta grafik performa.

Cara menjalankan:
    python train_model.py

Pastikan dataset sudah dikumpulkan terlebih dahulu dengan:
    python collect_dataset.py
"""

import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from neural_network import NeuralNetwork
from utils import load_dataset, one_hot_encode, split_dataset, ManualScaler

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


def plot_training_history(history, save_path=None):
    """
    Menampilkan grafik loss dan akurasi selama proses training.

    Grafik loss memperlihatkan seberapa cepat model belajar, sedangkan
    grafik akurasi menunjukkan seberapa baik model mengenali data
    training dan validasi dari waktu ke waktu.
    """
    if not PLT_AVAILABLE:
        print("[!] matplotlib tidak tersedia, grafik tidak ditampilkan")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', color='#FF6B6B', linewidth=2)
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss', color='#4ECDC4', linewidth=2)
    ax1.set_title('Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_accuracy'], label='Train Accuracy', color='#FF6B6B', linewidth=2)
    if history['val_accuracy']:
        ax2.plot(history['val_accuracy'], label='Val Accuracy', color='#4ECDC4', linewidth=2)
    ax2.set_title('Akurasi per Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Akurasi')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.suptitle('Training History - JST Backpropagation Hand Sign',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Grafik training disimpan ke: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, save_path=None):
    """
    Menampilkan confusion matrix sebagai heatmap.

    Diagonal utama menunjukkan prediksi yang benar. Nilai di luar
    diagonal menunjukkan pola kesalahan klasifikasi antar kelas.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    classes = [str(i) for i in range(cm.shape[0])]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix',
        ylabel='Label Sebenarnya',
        xlabel='Label Prediksi'
    )

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Confusion matrix disimpan ke: {save_path}")

    plt.show()


def main():
    print("=" * 65)
    print("  TRAINING JST BACKPROPAGATION - DETEKSI HAND SIGN ANGKA 0-9")
    print("=" * 65)
    print()

    # Langkah 1: Muat dataset dari CSV
    print("[1/5] Memuat dataset...")
    try:
        X, y = load_dataset()
    except FileNotFoundError:
        print("\n[ERROR] Dataset belum tersedia!")
        print("Jalankan dulu: python collect_dataset.py")
        return

    # Periksa apakah semua kelas sudah cukup sampelnya
    unique, counts = np.unique(y, return_counts=True)
    insufficient = [
        (label, count) for label, count in zip(unique, counts)
        if count < config.MIN_SAMPLES_PER_CLASS
    ]
    if insufficient:
        print("\n[WARNING] Beberapa kelas memiliki sampel kurang:")
        for label, count in insufficient:
            print(f"  Angka {label}: {count} sampel (minimum: {config.MIN_SAMPLES_PER_CLASS})")
        resp = input("\nLanjutkan training? (y/n): ").strip().lower()
        if resp != 'y':
            print("Training dibatalkan.")
            return

    # Langkah 2: Preprocessing data
    print("\n[2/5] Preprocessing data...")

    scaler   = ManualScaler()
    X_scaled = scaler.fit_transform(X)
    y_onehot = one_hot_encode(y)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_scaled, y_onehot,
        test_ratio=config.TEST_SPLIT,
        val_ratio=config.VALIDATION_SPLIT,
        seed=config.RANDOM_SEED
    )

    print(f"  Training  : {X_train.shape[0]} sampel")
    print(f"  Validasi  : {X_val.shape[0]} sampel")
    print(f"  Testing   : {X_test.shape[0]} sampel")

    # Langkah 3: Inisialisasi dan latih model
    print("\n[3/5] Memulai training...")

    model = NeuralNetwork(
        layer_sizes=config.NETWORK_ARCHITECTURE,
        learning_rate=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        reg_lambda=config.REGULARIZATION_LAMBDA,
        lr_decay=config.LEARNING_RATE_DECAY
    )
    model.summary()

    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True
    )

    # Langkah 4: Evaluasi model
    print("\n[4/5] Evaluasi model...")
    print("\nEvaluasi pada Training Set:")
    train_results = model.evaluate(X_train, y_train)
    print(f"  Akurasi: {train_results['accuracy'] * 100:.2f}%")

    print("\nEvaluasi pada Test Set:")
    test_results = model.evaluate(X_test, y_test)
    print(f"  Akurasi: {test_results['accuracy'] * 100:.2f}%")
    print(f"  Loss   : {test_results['loss']:.4f}")

    print("\n  Akurasi per kelas:")
    for cls, acc in test_results['per_class_accuracy'].items():
        print(f"    Angka {cls}: {acc * 100:.1f}%")

    # Tampilkan confusion matrix di terminal
    print("\n  Confusion Matrix:")
    cm     = test_results['confusion_matrix']
    header = "    P->  " + "  ".join([f"{i:3d}" for i in range(cm.shape[1])])
    print(header)
    print("    " + "=" * (len(header) - 4))
    for i in range(cm.shape[0]):
        row = f"  T {i} | " + "  ".join([f"{cm[i,j]:3d}" for j in range(cm.shape[1])])
        print(row)

    # Langkah 5: Simpan model dan grafik
    print("\n[5/5] Menyimpan model...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save_model(config.MODEL_PATH)
    scaler.save(config.SCALER_PATH)

    plot_path = os.path.join(config.MODEL_DIR, "training_history.png")
    cm_path   = os.path.join(config.MODEL_DIR, "confusion_matrix.png")
    plot_training_history(history, save_path=plot_path)
    plot_confusion_matrix(cm, save_path=cm_path)

    print("\n" + "=" * 65)
    print("  TRAINING SELESAI!")
    print("=" * 65)
    print(f"  Model   : {config.MODEL_PATH}")
    print(f"  Scaler  : {config.SCALER_PATH}")
    print(f"  Akurasi : {test_results['accuracy'] * 100:.2f}%")
    print()
    print("  Langkah selanjutnya:")
    print("    1. Test dengan gambar : python test_image.py <path_gambar>")
    print("    2. Test realtime      : python test_realtime.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
