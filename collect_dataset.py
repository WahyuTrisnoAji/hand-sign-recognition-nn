"""
MODUL PENGUMPULAN DATASET HAND SIGN
Program untuk merekam dataset hand sign angka 0-9 menggunakan webcam.
Menggunakan MediaPipe untuk deteksi landmark tangan.

Kontrol Keys:
  [0-9]  : Pilih angka yang akan direkam
  [SPACE] : Ambil 1 sampel
  [R]     : Toggle rekam otomatis (continuous recording)
  [D]     : Hapus sampel terakhir dari label aktif
  [C]     : Hapus semua data label aktif
  [S]     : Tampilkan statistik dataset
  [Q/ESC] : Keluar

Jalankan: python collect_dataset.py
"""

import cv2
import numpy as np
import csv
import os
import time
import sys

# Tambah parent dir ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import HandDetector, extract_landmarks, save_landmark_to_csv, get_dataset_stats


def draw_ui(frame, current_label, stats, recording, hand_detected, fps):
    """Gambar UI pada frame kamera."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel kiri: info dan statistik
    panel_w = 220
    cv2.rectangle(overlay, (0, 0), (panel_w, h), config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "HAND SIGN JST", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_CYAN, 2)
    cv2.putText(frame, "Dataset Recorder", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1)

    cv2.line(frame, (10, 65), (panel_w - 10, 65), config.COLOR_WHITE, 1)

    label_color = config.COLOR_RECORDING if recording else config.COLOR_READY
    cv2.putText(frame, f"Label Aktif: {current_label}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    status_text = "MEREKAM..." if recording else "SIAP"
    cv2.putText(frame, f"Status: {status_text}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    # Deteksi tangan
    hand_color = config.COLOR_GREEN if hand_detected else config.COLOR_RED
    hand_text = "Tangan: TERDETEKSI" if hand_detected else "Tangan: TIDAK ADA"
    cv2.putText(frame, hand_text, (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

    cv2.line(frame, (10, 175), (panel_w - 10, 175), config.COLOR_WHITE, 1)

    # Statistik per kelas
    cv2.putText(frame, "Dataset Stats:", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_YELLOW, 1)

    total = 0
    for i in range(config.NUM_CLASSES):
        count = stats.get(i, 0)
        total += count
        color = config.COLOR_GREEN if count >= config.MIN_SAMPLES_PER_CLASS else config.COLOR_WHITE
        if i == current_label:
            color = config.COLOR_YELLOW
            cv2.rectangle(frame, (5, 210 + i * 22), (panel_w - 5, 230 + i * 22),
                          (60, 60, 60), -1)

        bar_width = min(int(count / max(config.MIN_SAMPLES_PER_CLASS, 1) * 80), 80)
        cv2.rectangle(frame, (120, 213 + i * 22), (120 + bar_width, 227 + i * 22),
                      (0, 100, 0), -1)

        cv2.putText(frame, f"Angka {i}: {count:4d}", (10, 227 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    cv2.putText(frame, f"Total: {total}", (10, 227 + 10 * 22 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_CYAN, 1)

    cv2.rectangle(frame, (0, h - 45), (w, h), config.COLOR_PANEL_BG, -1)
    controls = "[0-9] Label | [SPACE] Ambil | [R] Rekam | [D] Hapus | [Q] Keluar"
    cv2.putText(frame, controls, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

    if recording:
        blink = int(time.time() * 3) % 2
        if blink:
            cv2.circle(frame, (w - 30, 30), 12, config.COLOR_RECORDING, -1)
            cv2.putText(frame, "REC", (w - 65, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_RECORDING, 2)

    if hand_detected:
        cv2.putText(frame, "OK", (w - 50, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_GREEN, 3)

    return frame


def delete_last_sample(csv_path, label):
    """Hapus sampel terakhir dengan label tertentu dari CSV."""
    if not os.path.isfile(csv_path):
        return False

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    for i in range(len(rows) - 1, 0, -1):
        if rows[i] and int(rows[i][0]) == label:
            rows.pop(i)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return True
    return False


def clear_label_data(csv_path, label):
    """Hapus semua data dengan label tertentu dari CSV."""
    if not os.path.isfile(csv_path):
        return 0

    rows = []
    removed = 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row != [] and (row[0] == 'label' or int(row[0]) != label):
                rows.append(row)
            else:
                removed += 1

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return removed


def main():
    """Program utama pengumpulan dataset."""
    print("=" * 60)
    print("  PENGUMPULAN DATASET HAND SIGN - JST BACKPROPAGATION")
    print("=" * 60)
    print()

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    detector = HandDetector()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera!")
        print("Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    current_label = 0
    recording = False
    record_interval = 0.15  # Interval rekam (detik)
    last_record_time = 0
    capture_flash = 0

    stats = get_dataset_stats()
    total_existing = sum(stats.values())
    if total_existing > 0:
        print(f"[Info] Dataset yang sudah ada: {total_existing} sampel")

    fps = 0
    frame_count = 0
    fps_start = time.time()

    print("\nKontrol:")
    print("  [0-9]   : Pilih angka yang akan direkam")
    print("  [SPACE] : Ambil 1 sampel")
    print("  [R]     : Toggle rekam otomatis")
    print("  [D]     : Hapus sampel terakhir")
    print("  [C]     : Hapus semua data label aktif")
    print("  [Q/ESC] : Keluar")
    print("\nMembuka kamera...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera")
            break

        frame = cv2.flip(frame, 1)  # Mirror

        # Deteksi tangan
        results = detector.detect(frame)
        hand_detected = results.multi_hand_landmarks is not None

        # Gambar landmark
        if hand_detected:
            frame = detector.draw_landmarks(frame, results)

        # Rekam otomatis
        if recording and hand_detected:
            current_time = time.time()
            if current_time - last_record_time >= record_interval:
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                save_landmark_to_csv(landmarks, current_label)
                stats[current_label] = stats.get(current_label, 0) + 1
                last_record_time = current_time
                capture_flash = 5
                total = sum(stats.values())
                print(f"\r  [REC] Angka {current_label} | "
                      f"Sampel: {stats[current_label]} | Total: {total}", end="")

        # Flash effect saat capture
        if capture_flash > 0:
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay, (0, 0),
                          (frame.shape[1], frame.shape[0]),
                          config.COLOR_GREEN, -1)
            alpha = capture_flash / 10.0
            cv2.addWeighted(flash_overlay, alpha, frame, 1 - alpha, 0, frame)
            capture_flash -= 1

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw UI
        frame = draw_ui(frame, current_label, stats, recording, hand_detected, fps)

        # Tampilkan
        cv2.imshow("Hand Sign Dataset Recorder - JST Backpropagation", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Angka 0-9
        if ord('0') <= key <= ord('9'):
            current_label = key - ord('0')
            print(f"\n  [Label] Beralih ke angka: {current_label}")

        # SPACE - ambil satu sampel
        elif key == ord(' '):
            if hand_detected:
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                save_landmark_to_csv(landmarks, current_label)
                stats[current_label] = stats.get(current_label, 0) + 1
                capture_flash = 8
                total = sum(stats.values())
                print(f"\n  [✓] Sampel diambil! Angka {current_label} | "
                      f"Count: {stats[current_label]} | Total: {total}")
            else:
                print("\n  [!] Tangan tidak terdeteksi. Tunjukkan tangan ke kamera.")

        # R - toggle recording
        elif key == ord('r') or key == ord('R'):
            recording = not recording
            if recording:
                print(f"\n  [REC] Rekam otomatis AKTIF untuk angka {current_label}")
            else:
                print(f"\n  [STOP] Rekam otomatis BERHENTI")

        # D - hapus sampel terakhir
        elif key == ord('d') or key == ord('D'):
            if delete_last_sample(config.DATASET_CSV, current_label):
                stats[current_label] = max(0, stats.get(current_label, 0) - 1)
                print(f"\n  [✗] Sampel terakhir angka {current_label} dihapus")
            else:
                print(f"\n  [!] Tidak ada sampel angka {current_label} untuk dihapus")

        # C - clear label data
        elif key == ord('c') or key == ord('C'):
            count = stats.get(current_label, 0)
            if count > 0:
                removed = clear_label_data(config.DATASET_CSV, current_label)
                stats[current_label] = 0
                print(f"\n  [✗] {removed} sampel angka {current_label} dihapus")
            else:
                print(f"\n  [!] Tidak ada data angka {current_label}")

        # S - show stats
        elif key == ord('s') or key == ord('S'):
            print("\n  " + "-" * 40)
            print("  STATISTIK DATASET:")
            total = 0
            for i in range(config.NUM_CLASSES):
                c = stats.get(i, 0)
                total += c
                bar = "█" * min(c // 5, 30)
                status = "✓" if c >= config.MIN_SAMPLES_PER_CLASS else "✗"
                print(f"    [{status}] Angka {i}: {c:4d} {bar}")
            print(f"    Total: {total}")
            print("  " + "-" * 40)

        # Q / ESC - keluar
        elif key == ord('q') or key == ord('Q') or key == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Ringkasan akhir
    print("\n" + "=" * 60)
    print("  RINGKASAN DATASET")
    print("=" * 60)
    stats = get_dataset_stats()
    total = 0
    ready = True
    for i in range(config.NUM_CLASSES):
        c = stats.get(i, 0)
        total += c
        status = "✓" if c >= config.MIN_SAMPLES_PER_CLASS else "✗"
        if c < config.MIN_SAMPLES_PER_CLASS:
            ready = False
        print(f"  [{status}] Angka {i}: {c:4d} sampel")
    print(f"\n  Total: {total} sampel")
    print(f"  File : {config.DATASET_CSV}")

    if ready:
        print("\n  [✓] Dataset SIAP untuk training!")
        print("  Jalankan: python train_model.py")
    else:
        print(f"\n  [!] Beberapa kelas belum mencapai minimum "
              f"({config.MIN_SAMPLES_PER_CLASS} sampel)")
        print("  Jalankan program ini lagi untuk menambah data.")

    print("=" * 60)


if __name__ == "__main__":
    main()
