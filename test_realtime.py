"""
Test Model JST secara Real-Time via Kamera

Modul ini menjalankan inferensi model JST secara langsung menggunakan
webcam. Setiap frame dianalisis untuk mendeteksi tangan, mengekstrak
fitur landmark, dan mengklasifikasikan gesture angka 0 hingga 9.

Hasil prediksi ditampilkan pada layar beserta bar probabilitas per kelas.
Jika fitur suara aktif, sistem akan mengucapkan angka yang terdeteksi
menggunakan text-to-speech.

Cara menjalankan:
    python test_realtime.py

Kontrol keyboard:
    V       : Aktifkan / matikan suara
    Q / ESC : Keluar dari program
"""

import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from neural_network import NeuralNetwork
from utils import HandDetector, extract_landmarks, ManualScaler, VoiceEngine


def draw_prediction_ui(frame, pred_class, confidence, probs, hand_detected, voice_on, fps):
    """
    Menggambar panel prediksi pada frame kamera.

    Panel kanan menampilkan angka prediksi, nilai confidence, dan
    bar probabilitas untuk setiap kelas. Panel bawah menampilkan
    status FPS, suara, dan kondisi tangan.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel kanan semi-transparan
    panel_x = w - 250
    cv2.rectangle(overlay, (panel_x, 0), (w, h), config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "JST HAND SIGN", (panel_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_CYAN, 2)
    cv2.putText(frame, "Real-Time Test", (panel_x + 10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1)
    cv2.line(frame, (panel_x + 10, 65), (w - 10, 65), config.COLOR_WHITE, 1)

    if hand_detected and pred_class is not None:
        # Angka prediksi ditampilkan besar di tengah panel
        color = (config.COLOR_GREEN if confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD
                 else config.COLOR_YELLOW)
        cv2.putText(frame, str(pred_class), (panel_x + 70, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 6)
        cv2.putText(frame, f"Conf: {confidence * 100:.1f}%", (panel_x + 30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Bar probabilitas per kelas
        cv2.putText(frame, "Probabilitas:", (panel_x + 10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1)

        for i in range(config.NUM_CLASSES):
            y_pos     = 240 + i * 22
            prob      = probs[i] if probs is not None else 0
            bar_color = (0, 200, 100) if i == pred_class else (100, 100, 100)
            bar_width = int(prob * 150)

            cv2.putText(frame, f"{i}", (panel_x + 10, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)
            cv2.rectangle(frame,
                          (panel_x + 30, y_pos),
                          (panel_x + 30 + bar_width, y_pos + 15),
                          bar_color, -1)
            cv2.rectangle(frame,
                          (panel_x + 30, y_pos),
                          (panel_x + 180, y_pos + 15),
                          (80, 80, 80), 1)

            if prob > 0.05:
                cv2.putText(frame, f"{prob * 100:.0f}%",
                            (panel_x + 185, y_pos + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, config.COLOR_WHITE, 1)
    else:
        # Pesan panduan saat tangan belum terdeteksi
        cv2.putText(frame, "Tunjukkan", (panel_x + 30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 1)
        cv2.putText(frame, "Hand Sign",  (panel_x + 35, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 1)
        cv2.putText(frame, "ke Kamera",  (panel_x + 35, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 1)

    # Panel status bawah
    cv2.rectangle(frame, (0, h - 40), (w, h), config.COLOR_PANEL_BG, -1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1)

    voice_text  = "Suara: ON"  if voice_on else "Suara: OFF"
    voice_color = config.COLOR_GREEN if voice_on else config.COLOR_RED
    cv2.putText(frame, voice_text, (120, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, voice_color, 1)

    hand_text  = "Tangan: OK"   if hand_detected else "Tangan: --"
    hand_color = config.COLOR_GREEN if hand_detected else config.COLOR_RED
    cv2.putText(frame, hand_text, (280, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)

    cv2.putText(frame, "[V] Suara | [Q] Keluar", (w - 250, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

    return frame


def main():
    print("=" * 60)
    print("  TEST REAL-TIME JST - DETEKSI HAND SIGN VIA KAMERA")
    print("=" * 60)

    if not os.path.isfile(config.MODEL_PATH):
        print("\n[ERROR] Model belum tersedia!")
        print("Jalankan dulu:")
        print("  1. python collect_dataset.py  (kumpulkan dataset)")
        print("  2. python train_model.py      (latih model)")
        return

    model = NeuralNetwork.load_model(config.MODEL_PATH)
    model.summary()

    scaler = ManualScaler()
    if os.path.isfile(config.SCALER_PATH):
        scaler.load(config.SCALER_PATH)

    detector  = HandDetector()
    voice     = VoiceEngine(enabled=config.VOICE_ENABLED)
    voice_on  = config.VOICE_ENABLED

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # Variabel untuk stabilitas prediksi sebelum suara dibunyikan
    pred_class       = None
    confidence       = 0
    probs            = None
    last_pred        = -1
    stable_count     = 0
    stable_threshold = 5    # Jumlah frame konsisten sebelum suara pertama

    fps         = 0
    frame_count = 0
    fps_start   = time.time()

    print("\nKontrol:")
    print("  [V]     : Toggle suara on/off")
    print("  [Q/ESC] : Keluar")
    print("\nMembuka kamera...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results      = detector.detect(frame)
        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            frame = detector.draw_landmarks(frame, results)

            landmarks         = extract_landmarks(results.multi_hand_landmarks[0])
            landmarks_scaled  = scaler.transform(landmarks.reshape(1, -1))
            pred_class, confidence, probs = model.predict_single(landmarks_scaled)

            # Hitung stabilitas: berapa frame berturut-turut prediksi sama
            if pred_class == last_pred:
                stable_count += 1
            else:
                # Gesture baru: reset dan izinkan suara langsung muncul
                stable_count = 0
                last_pred    = pred_class
                voice.reset()

            # Suara dibunyikan hanya jika prediksi stabil dan confidence cukup
            if voice_on and stable_count >= stable_threshold:
                if confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD:
                    voice.speak_number(pred_class)
        else:
            # Reset semua state saat tangan hilang dari kamera
            pred_class   = None
            confidence   = 0
            probs        = None
            stable_count = 0
            last_pred    = -1
            voice.reset()

        # Hitung FPS setiap satu detik
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            fps_start   = time.time()

        frame = draw_prediction_ui(frame, pred_class, confidence, probs,
                                   hand_detected, voice_on, fps)
        cv2.imshow("Hand Sign Detection - JST Backpropagation (Real-Time)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('v'), ord('V')):
            voice_on = not voice_on
            status   = "ON" if voice_on else "OFF"
            print(f"  [Suara] {status}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nSelesai.")


if __name__ == "__main__":
    main()
