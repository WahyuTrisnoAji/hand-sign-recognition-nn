"""
Implementasi Jaringan Syaraf Tiruan (JST) dengan Backpropagation

Modul ini berisi implementasi JST dari awal menggunakan NumPy, tanpa
ketergantungan pada framework deep learning seperti TensorFlow atau
PyTorch. Seluruh proses forward propagation, fungsi loss, backpropagation,
dan update bobot ditulis secara eksplisit.

Metode Optimasi:
    Mini-Batch Stochastic Gradient Descent (SGD) dengan Momentum,
    dilengkapi L2 Regularization dan Dropout pada hidden layer.

Referensi Matematis:
    Forward  : z[l] = a[l-1] · W[l] + b[l],  a[l] = f(z[l])
    Loss     : L = -1/m * sum(sum(y * log(y_hat)))
    Backward : delta[L] = y_hat - y  (turunan Softmax + Cross-Entropy)
               delta[l] = delta[l+1] · W[l+1].T * f'(z[l])
    Update   : W[l] -= lr * (1/m * a[l-1].T · delta[l] + lambda * W[l])
"""

import numpy as np
import os
import time


class NeuralNetwork:
    """
    Jaringan Syaraf Tiruan dengan algoritma Backpropagation.

    Arsitektur jaringan didefinisikan sebagai daftar ukuran neuron per
    layer. Hidden layer menggunakan aktivasi ReLU, sedangkan output
    layer menggunakan Softmax. Dropout diterapkan hanya pada hidden
    layer selama proses training.

    Parameters
    ----------
    layer_sizes : list
        Jumlah neuron tiap layer, misalnya [86, 256, 128, 64, 10].
    learning_rate : float
        Laju pembelajaran awal.
    momentum : float
        Koefisien momentum untuk mempercepat konvergensi SGD.
    reg_lambda : float
        Bobot L2 regularization untuk mencegah overfitting.
    lr_decay : float
        Faktor peluruhan learning rate yang diterapkan setiap epoch.
    dropout_rate : float
        Proporsi neuron yang dinonaktifkan saat training (inverted dropout).
    """

    def __init__(self, layer_sizes, learning_rate=0.001, momentum=0.9,
                 reg_lambda=0.0001, lr_decay=0.999, dropout_rate=0.15):
        self.layer_sizes    = layer_sizes
        self.n_layers       = len(layer_sizes)
        self.learning_rate  = learning_rate
        self.initial_lr     = learning_rate
        self.momentum       = momentum
        self.reg_lambda     = reg_lambda
        self.lr_decay       = lr_decay
        self.dropout_rate   = dropout_rate
        self._training_mode = False     # True saat training, False saat inferensi

        self.weights = []
        self.biases  = []
        self._initialize_weights()

        # Velocity untuk SGD Momentum
        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]

        # Menyimpan nilai aktivasi dan pre-aktivasi untuk backpropagation
        self.activations = []
        self.z_values    = []

        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss':   [], 'val_accuracy':   []
        }

    def _initialize_weights(self):
        """
        Inisialisasi bobot menggunakan metode He Initialization.

        He Init cocok untuk aktivasi ReLU karena mempertahankan
        variansi sinyal di seluruh lapisan jaringan yang dalam.
        Formula: W ~ N(0, sqrt(2 / n_in))
        """
        np.random.seed(42)
        self.weights = []
        self.biases  = []
        for i in range(self.n_layers - 1):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)


    # ======================================================================
    # Fungsi Aktivasi
    # ======================================================================

    @staticmethod
    def relu(z):
        """ReLU: f(z) = max(0, z). Menghasilkan 0 untuk nilai negatif."""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """Turunan ReLU: 1 jika z > 0, dan 0 jika sebaliknya."""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        """
        Softmax untuk output layer, mengubah logit menjadi distribusi probabilitas.

        Menggunakan trik numerik z - max(z) agar eksponensiasi tidak overflow
        saat nilai z sangat besar.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    # ======================================================================
    # Forward Propagation
    # ======================================================================

    def forward(self, X):
        """
        Menghitung output jaringan dari input X melalui semua layer.

        Dropout diterapkan pada setiap hidden layer saat mode training
        aktif (_training_mode=True). Saat inferensi, semua neuron aktif
        dan bobot tidak perlu discale ulang karena menggunakan inverted dropout.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Batch data input.

        Returns
        -------
        np.ndarray, shape (m, n_classes)
            Probabilitas prediksi dari softmax.
        """
        self.activations    = [X]
        self.z_values       = []
        self._dropout_masks = []    # Simpan mask agar bisa dipakai ulang di backward

        a = X

        # Hidden layers: aktivasi ReLU dengan Dropout saat training
        for i in range(self.n_layers - 2):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)

            if self._training_mode and self.dropout_rate > 0:
                # Inverted dropout: skala agar ekspektasi output tetap sama
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                mask /= (1.0 - self.dropout_rate)
                a    = a * mask
                self._dropout_masks.append(mask)
            else:
                self._dropout_masks.append(None)

            self.activations.append(a)

        # Output layer: Softmax tanpa dropout
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)

        return a


    # ======================================================================
    # Fungsi Loss
    # ======================================================================

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Menghitung Categorical Cross-Entropy Loss ditambah L2 Regularization.

        Formula:
            L = -1/m * sum(y_true * log(y_pred)) + lambda/(2m) * sum(W^2)

        Epsilon kecil ditambahkan ke y_pred untuk mencegah log(0).

        Parameters
        ----------
        y_pred : np.ndarray, shape (m, n_classes)
            Output probabilitas dari softmax.
        y_true : np.ndarray, shape (m, n_classes)
            Label sebenarnya dalam format one-hot encoding.
        """
        m       = y_true.shape[0]
        epsilon = 1e-8
        ce_loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m

        l2_loss = sum(np.sum(w ** 2) for w in self.weights)
        l2_loss = (self.reg_lambda / (2 * m)) * l2_loss

        return ce_loss + l2_loss


    # ======================================================================
    # Backpropagation
    # ======================================================================

    def backward(self, y_true):
        """
        Menghitung gradien error dan memperbarui seluruh bobot jaringan.

        Langkah-langkah:
            1. Hitung error di output layer: delta = y_hat - y
               (Ini adalah turunan gabungan Softmax dan Cross-Entropy)
            2. Propagasi balik error ke hidden layer menggunakan chain rule
            3. Hitung gradien bobot: dW = (1/m) * a_prev.T · delta
            4. Perbarui bobot dengan SGD Momentum

        Parameters
        ----------
        y_true : np.ndarray, shape (m, n_classes)
            Label sebenarnya dalam format one-hot encoding.
        """
        m = y_true.shape[0]

        # Langkah 1: Error di output layer
        delta  = self.activations[-1] - y_true
        deltas = [delta]

        # Langkah 2: Propagasi balik melalui hidden layer
        for i in range(self.n_layers - 3, -1, -1):
            d = (deltas[-1] @ self.weights[i + 1].T) * \
                    self.relu_derivative(self.z_values[i])

            # Terapkan dropout mask yang sama seperti saat forward pass
            mask = self._dropout_masks[i] if hasattr(self, '_dropout_masks') else None
            if mask is not None:
                d = d * mask
            deltas.append(d)

        deltas.reverse()

        # Langkah 3 & 4: Hitung gradien dan perbarui bobot dengan Momentum
        for i in range(self.n_layers - 1):
            dW = (self.activations[i].T @ deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Tambahkan gradien L2 regularization
            dW += (self.reg_lambda / m) * self.weights[i]

            # Update dengan SGD Momentum: v = mu*v - lr*dW
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.learning_rate * dW
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.learning_rate * db

            self.weights[i] += self.vel_w[i]
            self.biases[i]  += self.vel_b[i]


    # ======================================================================
    # Training
    # ======================================================================

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=500, batch_size=32, patience=50, verbose=True):
        """
        Melatih jaringan dengan Mini-Batch SGD, Momentum, dan Early Stopping.

        Setiap epoch, data training dikocok (shuffle) sebelum dibagi menjadi
        mini-batch. Setelah setiap epoch, model dievaluasi pada data validasi.
        Jika loss validasi tidak membaik selama 'patience' epoch, training
        dihentikan dan bobot terbaik dipulihkan.

        Parameters
        ----------
        X_train, y_train : data training dan label (one-hot).
        X_val, y_val     : data validasi dan label (optional).
        epochs           : jumlah epoch maksimal.
        batch_size       : ukuran mini-batch.
        patience         : epoch toleransi tanpa perbaikan (early stopping).
        verbose          : tampilkan log progres ke terminal.

        Returns
        -------
        dict
            Riwayat loss dan akurasi per epoch (train dan validasi).
        """
        m              = X_train.shape[0]
        best_val_loss  = float('inf')
        patience_counter = 0
        best_weights   = None
        best_biases    = None

        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss':   [], 'val_accuracy':   []
        }

        if verbose:
            print("=" * 65)
            print("TRAINING JARINGAN SYARAF TIRUAN - BACKPROPAGATION")
            print("=" * 65)
            print(f"Arsitektur    : {self.layer_sizes}")
            print(f"Learning Rate : {self.learning_rate}")
            print(f"Momentum      : {self.momentum}")
            print(f"Batch Size    : {batch_size}")
            print(f"Max Epochs    : {epochs}")
            print(f"Training Data : {m} sampel")
            if X_val is not None:
                print(f"Validation    : {X_val.shape[0]} sampel")
            print("=" * 65)

        start_time = time.time()

        for epoch in range(epochs):
            self._training_mode = True

            # Kocok data training sebelum dibagi ke mini-batch
            indices    = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for j in range(0, m, batch_size):
                X_batch = X_shuffled[j:j + batch_size]
                y_batch = y_shuffled[j:j + batch_size]
                self.forward(X_batch)
                self.backward(y_batch)

            # Peluruhan learning rate setiap epoch
            self.learning_rate = self.initial_lr * (self.lr_decay ** epoch)

            # Evaluasi dalam mode inferensi (dropout dimatikan)
            self._training_mode = False
            y_pred_train = self.forward(X_train)
            train_loss   = self.cross_entropy_loss(y_pred_train, y_train)
            train_acc    = np.mean(
                np.argmax(y_pred_train, axis=1) == np.argmax(y_train, axis=1)
            )

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss   = self.cross_entropy_loss(y_pred_val, y_val)
                val_acc    = np.mean(
                    np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1)
                )
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                # Simpan bobot terbaik berdasarkan validation loss
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    best_weights     = [w.copy() for w in self.weights]
                    best_biases      = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                          f"LR: {self.learning_rate:.6f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"\n[Early Stopping] Berhenti di epoch {epoch + 1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                          f"LR: {self.learning_rate:.6f}")

        # Pulihkan bobot terbaik yang tersimpan
        if best_weights is not None:
            self.weights = best_weights
            self.biases  = best_biases
            if verbose:
                print(f"[Info] Menggunakan bobot terbaik (val_loss: {best_val_loss:.4f})")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTraining selesai dalam {elapsed:.1f} detik")
            final_pred = self.forward(X_train)
            final_acc  = np.mean(
                np.argmax(final_pred, axis=1) == np.argmax(y_train, axis=1)
            )
            print(f"Akurasi akhir (train): {final_acc:.4f} ({final_acc * 100:.1f}%)")

        return self.history


    # ======================================================================
    # Prediksi
    # ======================================================================

    def predict(self, X):
        """Mengembalikan probabilitas prediksi untuk setiap kelas."""
        return self.forward(X)

    def predict_class(self, X):
        """Mengembalikan indeks kelas dengan probabilitas tertinggi."""
        return np.argmax(self.forward(X), axis=1)

    def predict_single(self, x):
        """
        Prediksi untuk satu sampel tunggal.

        Returns
        -------
        tuple : (predicted_class, confidence, all_probabilities)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs      = self.forward(x)
        pred_class = np.argmax(probs, axis=1)[0]
        confidence = probs[0, pred_class]
        return pred_class, confidence, probs[0]


    # ======================================================================
    # Evaluasi Model
    # ======================================================================

    def evaluate(self, X, y_true):
        """
        Mengevaluasi performa model pada data yang diberikan.

        Menghitung akurasi keseluruhan, loss, confusion matrix, dan
        akurasi per kelas secara sekaligus.

        Returns
        -------
        dict
            Berisi 'accuracy', 'loss', 'confusion_matrix',
            'per_class_accuracy', 'predictions', dan 'true_labels'.
        """
        y_pred      = self.forward(X)
        loss        = self.cross_entropy_loss(y_pred, y_true)
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        accuracy    = np.mean(pred_classes == true_classes)

        # Buat confusion matrix
        n_classes = y_true.shape[1]
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(true_classes, pred_classes):
            cm[t, p] += 1

        # Akurasi per kelas
        per_class_acc = {}
        for c in range(n_classes):
            total = np.sum(true_classes == c)
            if total > 0:
                correct = np.sum((true_classes == c) & (pred_classes == c))
                per_class_acc[c] = correct / total
            else:
                per_class_acc[c] = 0.0

        return {
            'accuracy':          accuracy,
            'loss':              loss,
            'confusion_matrix':  cm,
            'per_class_accuracy': per_class_acc,
            'predictions':       pred_classes,
            'true_labels':       true_classes
        }


    # ======================================================================
    # Simpan dan Muat Model
    # ======================================================================

    def save_model(self, filepath):
        """
        Menyimpan seluruh parameter model (bobot, bias, arsitektur) ke file .npz.

        Format NPZ dipilih karena portabel, efisien, dan tidak memerlukan
        dependensi tambahan seperti pickle.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {
            'layer_sizes':   np.array(self.layer_sizes),
            'learning_rate': np.array([self.initial_lr]),
            'momentum':      np.array([self.momentum]),
            'reg_lambda':    np.array([self.reg_lambda]),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}']   = b
        np.savez(filepath, **save_dict)
        print(f"[OK] Model disimpan ke: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Memuat model dari file .npz dan merekonstruksi objek NeuralNetwork.

        Velocity diinisialisasi ulang ke nol karena tidak diperlukan
        saat inferensi.
        """
        data       = np.load(filepath)
        layer_sizes = data['layer_sizes'].tolist()
        lr         = float(data['learning_rate'][0])
        mom        = float(data['momentum'][0])
        reg        = float(data['reg_lambda'][0])

        model = cls(layer_sizes, learning_rate=lr, momentum=mom, reg_lambda=reg)
        n_weight_layers = len(layer_sizes) - 1
        model.weights   = [data[f'weight_{i}'] for i in range(n_weight_layers)]
        model.biases    = [data[f'bias_{i}']   for i in range(n_weight_layers)]
        model.vel_w     = [np.zeros_like(w) for w in model.weights]
        model.vel_b     = [np.zeros_like(b) for b in model.biases]

        print(f"[OK] Model dimuat dari: {filepath}")
        print(f"     Arsitektur: {layer_sizes}")
        return model

    def summary(self):
        """Menampilkan ringkasan arsitektur dan jumlah parameter tiap layer."""
        print("\n" + "=" * 50)
        print("RINGKASAN MODEL JST BACKPROPAGATION")
        print("=" * 50)
        total_params = 0
        for i in range(self.n_layers - 1):
            w_params     = self.weights[i].size
            b_params     = self.biases[i].size
            layer_params = w_params + b_params
            total_params += layer_params
            activation   = "ReLU" if i < self.n_layers - 2 else "Softmax"
            print(f"Layer {i}: {self.layer_sizes[i]:4d} -> {self.layer_sizes[i+1]:4d} "
                  f"({activation:7s}) | Params: {layer_params:,}")
        print("=" * 50)
        print(f"Total Parameter: {total_params:,}")
        print("=" * 50)


if __name__ == "__main__":
    # Contoh penggunaan sederhana untuk memverifikasi implementasi
    print("Demo JST Backpropagation")
    print("=" * 40)

    np.random.seed(42)
    X = np.random.randn(100, 63)
    y_indices = np.random.randint(0, 10, 100)
    y = np.zeros((100, 10))
    y[np.arange(100), y_indices] = 1

    model = NeuralNetwork([63, 128, 64, 10], learning_rate=0.01)
    model.summary()

    history = model.train(X, y, epochs=50, batch_size=16, verbose=True)

    pred_class, conf, probs = model.predict_single(X[0])
    print(f"\nPrediksi sampel pertama: kelas {pred_class} (confidence: {conf:.2f})")
