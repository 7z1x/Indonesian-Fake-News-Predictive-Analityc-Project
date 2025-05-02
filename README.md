# Laporan Proyek Machine Learning - Zulfahmi M. Ardianto

## Domain Proyek

### Latar Belakang

Berita palsu (fake news) telah menjadi isu global yang signifikan, terutama di Indonesia, negara dengan jumlah pengguna media sosial yang besar. Penyebaran informasi yang salah dapat menyebabkan kepanikan sosial, manipulasi opini publik, dan kerugian ekonomi. Penelitian seperti yang dilakukan oleh Kurniawan dan Mustikasari (2021) menunjukkan bahwa deteksi berita palsu dalam bahasa Indonesia memerlukan pendekatan khusus karena karakteristik bahasa dan konteks budaya yang unik. Oleh karena itu, pengembangan sistem otomatis untuk mengidentifikasi berita palsu sangat penting.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

-  Mengapa: Berita palsu dapat merusak kepercayaan publik dan memengaruhi pengambilan keputusan. Di Indonesia, dampaknya diperparah oleh tingginya penetrasi media sosial.

-  Bagaimana: Dengan menggunakan algoritma machine learning untuk mengklasifikasikan berita berdasarkan konten teksnya, sistem dapat membantu memfilter informasi yang tidak akurat secara efisien.


### **Informasi Data**

Sumber: Indonesia False News Dataset dari Kaggle.

- Jumlah Data: Kombinasi data latih dan uji, total X sampel (setelah penggabungan dan penghapusan missing value).

-  Kondisi Data: Berisi kolom judul (judul berita), narasi (isi berita), dan label (0 untuk asli, 1 untuk palsu). Beberapa baris memiliki missing value pada kolom label.

## Business Understanding

### Problem Statement:
Penyebaran berita palsu di Indonesia menyebabkan masalah seperti:
- Kepanikan sosial akibat informasi yang menyesatkan.
- Manipulasi opini publik yang dapat memengaruhi stabilitas sosial dan politik.
- Kerugian ekonomi akibat keputusan yang didasarkan pada informasi salah.
### Goals:
- Mengembangkan model machine learning yang dapat mengklasifikasikan berita dalam bahasa Indonesia sebagai asli atau palsu dengan akurasi tinggi.
- Memastikan model dapat menangani ketidakseimbangan kelas untuk mendeteksi berita palsu secara efektif.
### Solution Statements:
Proyek ini mengusulkan dua solusi:
- Support Vector Machine (SVM): Menggunakan TF-IDF untuk ekstraksi fitur dan hyperparameter tuning untuk mengoptimalkan parameter seperti C, kernel, dan gamma.
- Random Forest: Menggunakan pendekatan serupa dengan tuning parameter seperti n_estimators, max_depth, dan min_samples_split.

## Data Understanding

### **Fitur pada Dataset:**

## Data Preparation

## Modeling

## Evaluasi

## Kesimpulan

## Daftar Pustaka:

Kurniawan dan Mustikasari (2021) menggunakan CNN dan LSTM untuk deteksi berita palsu dalam bahasa Indonesia.

Rahutomo (2019) menunjukkan bahwa Naive Bayes efektif untuk klasifikasi berita hoax dengan fitur term frequency.

Fawaid et al. (2021) mencapai akurasi hingga 90% menggunakan Transformer Network pada dataset serupa.
