"""
Test Model JST dengan Input Gambar

Modul ini menguji model JST yang sudah dilatih menggunakan file gambar
hand sign. Tersedia dua mode operasi: mode command-line (path gambar
diberikan sebagai argumen) dan mode interaktif (user memilih file
dari prompt terminal).

Cara menjalankan:
    python test_image.py <path_gambar>
    python test_image.py              (mode interaktif)
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from neural_network import NeuralNetwork
from utils import extract_landmarks_from_image, ManualScaler, VoiceEngine

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


def test_single_image(image_path, model, scaler, voice=None):
    """
    Menguji model pada satu gambar hand sign.

    Mengekstrak landmark tangan dari gambar, menormalisasi fitur,
    melakukan prediksi dengan model JST, lalu menampilkan hasil
    prediksi beserta visualisasi probabilitas per kelas.

    Parameters
    ----------
    image_path : str
        Path ke file gambar yang akan diuji.
    model : NeuralNetwork
        Model JST yang sudah dilatih.
    scaler : ManualScaler
        Scaler untuk normalisasi fitur input.
    voice : VoiceEngine, optional
        Engine suara untuk mengucapkan hasil prediksi.
    """
    print(f"\n  Memproses: {image_path}")

    try:
        landmarks, annotated_img = extract_landmarks_from_image(image_path)
    except FileNotFoundError:
        print(f"  [ERROR] File tidak ditemukan: {image_path}")
        return
    except Exception as e:
        print(f"  [ERROR] Gagal memproses gambar: {e}")
        return

    if landmarks is None:
        print("  [!] Tangan tidak terdeteksi dalam gambar ini.")
        print("  Pastikan gambar menunjukkan hand sign dengan jelas.")

        if PLT_AVAILABLE:
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb)
            plt.title("Tangan Tidak Terdeteksi", fontsize=14)
            plt.axis('off')
            plt.show()
        return

    # Normalisasi fitur dan prediksi
    landmarks_scaled         = scaler.transform(landmarks.reshape(1, -1))
    pred_class, confidence, probs = model.predict_single(landmarks_scaled)

    print(f"\n  Angka Terdeteksi : {pred_class}")
    print(f"  Confidence       : {confidence * 100:.1f}%")

    print(f"\n  Probabilitas per kelas:")
    for i in range(config.NUM_CLASSES):
        bar_len = int(probs[i] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        marker  = " <-- prediksi" if i == pred_class else ""
        print(f"    Angka {i}: [{bar}] {probs[i] * 100:5.1f}%{marker}")

    # Ucapkan hasil prediksi jika confidence mencukupi
    if voice and confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD:
        voice.speak_number(pred_class)

    # Visualisasi dengan matplotlib
    if PLT_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Hand Sign Terdeteksi", fontsize=13)
        axes[0].axis('off')

        colors = ['#4ECDC4' if i != pred_class else '#FF6B6B'
                  for i in range(config.NUM_CLASSES)]
        bars = axes[1].barh(range(config.NUM_CLASSES), probs * 100, color=colors)
        axes[1].set_yticks(range(config.NUM_CLASSES))
        axes[1].set_yticklabels([f'Angka {i}' for i in range(config.NUM_CLASSES)])
        axes[1].set_xlabel('Probabilitas (%)')
        axes[1].set_title(f'Prediksi: Angka {pred_class} ({confidence * 100:.1f}%)',
                          fontsize=13, fontweight='bold')
        axes[1].set_xlim([0, 105])

        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                axes[1].text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                             f'{prob * 100:.1f}%', va='center', fontsize=9)

        plt.suptitle('Test Hand Sign - JST Backpropagation', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    print("=" * 60)
    print("  TEST MODEL JST - INPUT GAMBAR HAND SIGN")
    print("=" * 60)

    if not os.path.isfile(config.MODEL_PATH):
        print("\n[ERROR] Model belum tersedia!")
        print("Jalankan dulu: python train_model.py")
        return

    model = NeuralNetwork.load_model(config.MODEL_PATH)

    scaler = ManualScaler()
    if os.path.isfile(config.SCALER_PATH):
        scaler.load(config.SCALER_PATH)
    else:
        print("[WARNING] Scaler tidak ditemukan, data tidak ternormalisasi")

    voice = VoiceEngine(enabled=config.VOICE_ENABLED)

    # Mode command-line: path gambar diberikan sebagai argumen
    if len(sys.argv) > 1:
        for img_path in sys.argv[1:]:
            test_single_image(img_path, model, scaler, voice)
    else:
        # Mode interaktif: user memasukkan path atau nomor file
        print("\nMode Interaktif - Masukkan path gambar (ketik 'q' untuk keluar)")
        print("Atau letakkan gambar di folder:", config.TEST_IMAGES_DIR)

        if os.path.isdir(config.TEST_IMAGES_DIR):
            images = [f for f in os.listdir(config.TEST_IMAGES_DIR)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                print(f"\nGambar tersedia di test_images/:")
                for i, img in enumerate(images):
                    print(f"  [{i+1}] {img}")
                print()

        while True:
            user_input = input("\nMasukkan path gambar atau nomor file: ").strip()

            if user_input.lower() in ('q', 'quit', 'exit'):
                break

            # Jika user memasukkan nomor, konversi ke path file
            if user_input.isdigit():
                idx = int(user_input) - 1
                if os.path.isdir(config.TEST_IMAGES_DIR):
                    images = [f for f in os.listdir(config.TEST_IMAGES_DIR)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if 0 <= idx < len(images):
                        user_input = os.path.join(config.TEST_IMAGES_DIR, images[idx])
                    else:
                        print(f"  [!] Nomor tidak valid. Pilih 1 hingga {len(images)}")
                        continue

            if os.path.isfile(user_input):
                test_single_image(user_input, model, scaler, voice)
            else:
                print(f"  [!] File tidak ditemukan: {user_input}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
