#Autoencoder untuk Denoising & Sharpening Gambar pada Dataset Facades

Proyek ini bertujuan untuk membangun dan melatih model **Autoencoder Convolutional Neural Network (CNN)** dengan menggunakan dataset *Facades* dari Kaggle. Model ini digunakan untuk merekonstruksi gambar input dan meningkatkan ketajaman hasilnya melalui filter sharpening.

---

## 📦 Dataset

Dataset digunakan dari Kaggle: [`balraj98/facades-dataset`](https://www.kaggle.com/datasets/balraj98/facades-dataset)

Dataset diunduh menggunakan `kagglehub` dan semua gambar diekstrak ke folder `frames`.

---

## 🏗️ Arsitektur Proyek

### 1.Preprocessing dan Loading Dataset
- Semua gambar `.jpg` dan `.png` dari dataset dipindahkan ke folder `frames`.
- Gambar diresize menjadi `(128, 128)` dan dikonversi ke tensor PyTorch menggunakan `torchvision.transforms`.

### 2.Custom Dataset Loader
- Kelas `CustomImageDataset` dibuat untuk me-load data dari folder.
- Setiap gambar dikembalikan dua kali (`return image, image`) karena digunakan untuk training autoencoder (input dan target sama).

### 3. Model: Autoencoder
- Arsitektur terdiri dari:
  - **Encoder:** 3 layer `Conv2d` dengan `ReLU`, secara bertahap mengurangi dimensi spasial dan meningkatkan channel.
  - **Decoder:** 3 layer `ConvTranspose2d` dengan `ReLU` dan `Sigmoid`, mengembalikan gambar ke resolusi awal.

### 4. **Fungsi Sharpening**
- Setelah output dihasilkan, dilakukan sharpening menggunakan kernel konvolusi sederhana:
  ```
  [[ 0, -1,  0],
   [-1,  5, -1],
   [ 0, -1,  0]]
  ```

### 5. **Evaluasi dengan PSNR**
- Menggunakan fungsi `calculate_psnr()` untuk mengukur kualitas hasil rekonstruksi terhadap input asli.

### 6. **Training Loop**
- Model dilatih selama 5 epoch.
- Loss function: `Mean Squared Error (MSE)`.
- Optimizer: `Adam` dengan `lr = 1e-3`.
- Pada akhir tiap epoch:
  - Disimpan gambar input dan output yang sudah di-sharpen ke folder `results/`.
  - Dihitung dan dicetak nilai *Loss* dan *PSNR*.

### 7. **Visualisasi**
- Digunakan `matplotlib` untuk:
  - Menampilkan grafik loss dan PSNR per epoch.
  - Menampilkan beberapa contoh gambar input dan hasil rekonstruksi.

---

## 🖼️ Struktur Folder

```
project/
│
├── frames/                  # Folder penyimpanan gambar dari dataset
├── results/                 # Folder penyimpanan output model per epoch
│   ├── input_epoch{n}_img{i}.png
│   ├── output_epoch{n}_img{i}.png
│   └── metrics.png          # Grafik loss & PSNR
│
├── main.py                  # Script utama (isi dari kode di atas)
└── README.md                # File penjelasan ini
```

---

## 💻 Requirements

Install dependencies berikut sebelum menjalankan kode:

```bash
pip install torch torchvision kagglehub matplotlib pillow
```

---

## 🚀 Cara Menjalankan

1. Pastikan kamu memiliki kredensial Kaggle (`kaggle.json`) yang valid di environment-mu.
2. Jalankan script `main.py`:
```bash
python main.py
```

---

## 📈 Contoh Output

- Gambar hasil rekonstruksi dan sharpening dapat dilihat di folder `results/`.
- Grafik `metrics.png` menunjukkan perkembangan nilai *Loss* dan *PSNR* selama training.

---

## ✍️ Catatan

- Dataset digunakan untuk melatih autoencoder secara **unsupervised**.
- Fungsi sharpening ditambahkan sebagai langkah pasca-pemrosesan untuk meningkatkan tampilan visual dari hasil rekonstruksi.
- Dataset *facades* cocok untuk eksperimen sederhana dengan model autoencoder karena strukturnya yang konsisten.
