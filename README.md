# Laporan Proyek Machine Learning - Zulfahmi M. Ardianto

## Domain Proyek

### Latar Belakang

Berita palsu (fake news) telah menjadi isu global yang signifikan, terutama di Indonesia, negara dengan jumlah pengguna media sosial yang besar. Penyebaran informasi yang salah dapat menyebabkan kepanikan sosial, manipulasi opini publik, dan kerugian ekonomi. Deteksi berita palsu dalam bahasa Indonesia memerlukan pendekatan khusus karena karakteristik bahasa dan konteks budaya yang unik. Oleh karena itu, pengembangan sistem otomatis untuk mengidentifikasi berita palsu sangat penting.[1]

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

-  Mengapa: Berita palsu dapat merusak kepercayaan publik dan memengaruhi pengambilan keputusan. Di Indonesia, dampaknya diperparah oleh tingginya penetrasi media sosial.

-  Bagaimana: Dengan menggunakan algoritma machine learning untuk mengklasifikasikan berita berdasarkan konten teksnya, sistem dapat membantu memfilter informasi yang tidak akurat secara efisien.

## Business Understanding

### Problem Statement:
- Bagaimana cara mengidentifikasi apakah suatu berita tergolong hoax atau tidak secara otomatis?
### Goals:
- Mengembangkan model machine learning yang dapat mengklasifikasikan berita dalam bahasa Indonesia sebagai asli atau palsu dengan akurasi tinggi.
- Memastikan model dapat menangani ketidakseimbangan kelas untuk mendeteksi berita palsu secara efektif.
### Solution Statements:
Proyek ini mengusulkan dua solusi:
- Support Vector Machine (SVM): Menggunakan TF-IDF untuk ekstraksi fitur dan hyperparameter tuning untuk mengoptimalkan parameter seperti C, kernel, dan gamma.
- Random Forest: Menggunakan pendekatan serupa dengan tuning parameter seperti n_estimators, max_depth, dan min_samples_split.

## Data Understanding
### **Informasi Data**

Sumber: Indonesia False News Dataset dari [Kaggle](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews) dan bersumber dari akun kaggle yaitu Muhammad Ghazi Muharam.

- Jumlah Data: Kombinasi data latih dan uji, berjumlah total 4701 dan total 4231 sampel (setelah penggabungan dan penghapusan missing value).
  
### **Fitur pada Dataset:**
- Data set ini berjumlah 4701 baris data, dan memiliki 6 fitur yang terdiri dari ID, label, tanggal, judul, narasi, nama file gambar. Berikut adalah pengecekan jumlah baris dan kolom pada dataset dengan menggunakan memanggil variabel data yang telah dideklarasikan.
 
![](https://github.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/blob/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/adta%20distirbusi.jpg)

- Terdapat missing value (NAN) atau nilai kosong pada kolom label yang perlu penanganan, Berikut tabelnya:
  
  | Kolom           	| Jumlah NaN 	|
	|-----------------	|------------	|
	| ID           	    | 0          	|
	| label             | 470         |
	| tanggal  	        | 0          	|
	| judul      	      | 0         	|
	| narasi          	| 0         	|
	| nama file gambar	| 0          	|

- Tahap ini dilakukan untuk menghapus kolom yang mengandung nilai kosong (NaN) guna memastikan kualitas data sebelum masuk ke tahap pemodelan.
  
### **Distribusi data label:**
| Label | Nama         | Jumlah |
|-------|--------------|--------|
| 1     | Berita Hoax  | 3465   |
| 0     | Berita Benar | 766    |

![](https://github.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/blob/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/distribusi%20data%20label.jpg)

- Menganalisis panjang teks untuk melihat apakah ada pola tertentu antara panjang teks dan label (hoax/benar).
![](https://github.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/blob/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/panjang%20teks.jpg)


## Data Preparation
- ***Menggabungan Kolom***: Menggabungkan judul dan narasi menjadi kolom teks.
![](https://raw.githubusercontent.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/data%20yang%20sudah%20menggabungkan%20judul%20dan%20narasi%20lalu%20di%20tambahkan%C2%A0label.jpg)

### Preprocessing Teks:
- Konversi ke huruf kecil.<br>
- Penghapusan angka menggunakan regex.<br>
- Penghapusan tanda baca menggunakan string.punctuation.<br>
- Penghapusan stopwords dengan Sastrawi.<br>
- Stemming dengan Sastrawi untuk mengurangi kata ke bentuk dasar.<br>

- ***Train-Test-Split***<br>
Data yang digunakan adalah teks berita yang sudah dibersihkan dan label yang menunjukkan apakah berita tersebut hoax (1) atau benar (0). Pembagian dilakukan dengan proporsi 80% untuk data latih dan 20% untuk data uji, sambil menjaga agar jumlah berita hoax dan benar tetap seimbang di kedua bagian. Setelah itu, kode akan mencetak jumlah data latih dan data uji yang berhasil dipisahkan.<br>
- ***Ekstrasi Fitur***<br>
Digunakan untuk mengubah teks berita menjadi angka agar bisa diproses oleh model machine learning. Caranya adalah dengan menggunakan TF-IDF (Term Frequency–Inverse Document Frequency), yaitu metode yang mengukur seberapa penting sebuah kata dalam suatu dokumen dibandingkan dengan seluruh kumpulan dokumen. Di sini digunakan teknik unigram dan bigram (kombinasi satu atau dua kata berurutan) dengan maksimal **1000 fitur** atau kata penting. Data latih diolah terlebih dahulu untuk membentuk pola, lalu pola itu digunakan untuk mengubah data uji ke bentuk yang sama.<br>

- ***Penanganan Ketidakseimbangan Kelas:*** Menggunakan SMOTE untuk menyeimbangkan jumlah sampel antar kelas.<br>
Teknik yang digunakan adalah SMOTE (Synthetic Minority Over-sampling Technique), yaitu metode yang secara otomatis membuat data sintetis (tiruan) untuk menambah jumlah data di kelas minoritas. Dalam hal ini, SMOTE diterapkan pada data latih (X_train dan y_train) yang sudah berbentuk angka dari hasil TF-IDF.

Distribusi kelas setelah SMOTE:
| Label        | Jumlah     |
|--------------|------------|
| 0 (Benar)    | 2771       |
| 1 (Hoax)     | 2771       |


## Modeling
Dalam proyek ini, digunakan dua algoritma machine learning populer untuk melakukan klasifikasi berita hoax, yaitu Support Vector Machine (SVM) dan Random Forest Classifier. Keduanya dipilih karena memiliki karakteristik yang sesuai untuk menangani data teks serta mampu menghasilkan performa yang baik pada kasus klasifikasi biner.
- ***Support Vector Machine (SVM):***<br>
Algoritma ini bekerja dengan mencari hyperplane terbaik yang dapat memisahkan kelas secara optimal. Dalam proyek ini, SVM digunakan dengan kernel linear, yang sangat sesuai untuk teks dan data sparsity yang tinggi. Selain itu, dilakukan tuning pada parameter seperti C untuk mengatur margin regularisasi, sehingga model tidak overfitting terhadap data latih.
- ***Random Forest Classifier***<br>
Random Forest adalah algoritma berbasis ensemble learning yang terdiri dari banyak decision tree. Model ini secara otomatis melakukan voting dari banyak pohon keputusan, sehingga lebih tahan terhadap overfitting dan noise pada data. Random Forest sangat cocok digunakan untuk membandingkan performa dengan SVM karena mampu menangkap pola kompleks secara non-linear. Parameter seperti n_estimators (jumlah pohon) dan max_depth (kedalaman maksimum pohon) disesuaikan untuk mendapatkan performa terbaik

Berikut adalah hasil akurasi dari kedua model:

| Model                 	| *Accuracy* 	|
|-----------------------	|----------		|
| *SVM* 	                | 0.8417    	|
| *Random Forest*       	| 0.8370    	|

## Evaluasi
Setelah hyperparameter tuning, model dengan akurasi tertinggi pada data uji dipilih. SVM sering kali unggul dalam tugas klasifikasi teks, sehingga kemungkinan menjadi pilihan terbaik jika metrik mendukung.

***Metrik Evaluasi***<br>
| Metrik      | Rumus                                      | Penjelasan                                                                                  |
|-------------|--------------------------------------------|---------------------------------------------------------------------------------------------|
| Akurasi     | (TP + TN) / (TP + TN + FP + FN)            | Mengukur proporsi prediksi benar; memberikan gambaran umum performa model.                  |
| Presisi     | TP / (TP + FP)                             | Mengukur ketepatan prediksi positif.                                                        |
| Recall      | TP / (TP + FN)                             | Mengukur kemampuan mendeteksi positif sebenarnya; penting untuk kelas minoritas.            |
| F1-score    | 2 * (Presisi * Recall) / (Presisi + Recall)| Menyeimbangkan presisi dan recall; cocok untuk dataset yang tidak seimbang.                 |

- Confusion Matrix: Menampilkan TP, TN, FP, FN.<br>

| Confusion Matrix      | Penjelasan                                                                 |
|-----------------------|----------------------------------------------------------------------------|
| *True Positive (TP)*  | Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya |
| *False Positive (FP)* | Jumlah prediksi positif yang salah                                         |
| *True Negative (TN)*  | Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya |
| *False Negative (FN)* | Jumlah prediksi negatif yang salah                                         |
- Cross-validation:
Cross-validation adalah teknik evaluasi model machine learning yang bertujuan untuk mengukur seberapa baik model akan bekerja pada data yang belum pernah dilihat sebelumnya. Metode ini membantu menghindari overfitting (model terlalu cocok dengan data latih) dan memberikan gambaran performa model yang lebih stabil.

***Ringkasan Evaluasi***<br>
- Setelah hypertuning menggunakan RandomizedSearchCV<br>
RandomizedSearchCV adalah teknik pencarian hyperparameter yang digunakan untuk menemukan kombinasi parameter terbaik dari sebuah model machine learning.<br>

| Model         | Kelas | Accuracy | Precision | Recall | F1-Score | Cross-Validation Score |
|---------------|-------|----------|-----------|--------|----------|-------------------------|
| SVM           | 0     | 0.8417   | 0.71      | 0.21   | 0.32     | 0.9264                  |
|               | 1     |          | 0.85      | 0.98   | 0.91     |                         |
| Random Forest | 0     | 0.8394   | 0.58      | 0.42   | 0.48     | 0.9142                  |
|               | 1     |          | 0.88      | 0.93   | 0.90     |                         |


- Confusion Matrix Random Forest<br>
![](https://github.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/blob/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/WhatsApp%20Image%202025-05-02%20at%2019.25.54_b7434aba.jpg)

- Confusion Matrix Support Vector Machine<br>
![](https://github.com/7z1x/Indonesian-Fake-News-Predictive-Analityc-Project/blob/6ef8af1438317e0b8f10ea3e96763a8ea924d6c6/image/svm%20.jpg)

## Kesimpulan
- SVM adalah model terbaik dari dua yang diuji, unggul dalam hampir semua metrik terutama pada deteksi berita hoax.
- Performa di Kelas 0 (Berita Benar) Buruk: Recall hanya 0.21 menunjukkan banyak berita benar salah diklasifikasikan sebagai hoax.
- Kurang Seimbang: SVM tampak terlalu fokus mengenali hoax, sehingga mengorbankan akurasi kelas lainnya. Ini kemungkinan karena imbalance class, meskipun sudah diatasi dengan SMOTE.
- Proyek berhasil membangun sistem yang sangat sensitif terhadap berita hoax, namun masih perlu peningkatan agar tidak terlalu bias terhadap kelas hoax saja.
- Ke depannya, mungkin perlu strategi untuk menyeimbangkan performa antar kelas, misalnya dengan penyesuaian threshold atau teknik balancing tambahan, dan juga mungkin menggunakan model yang lain yang lebih efektif seperti LSTM dan GRU.

## Daftar Pustaka:

[1] Kurniawan, A. A., & Mustikasari, M. (2021). Implementasi deep learning menggunakan metode CNN dan LSTM untuk menentukan berita palsu dalam bahasa Indonesia. Jurnal Informatika Universitas Pamulang, 5(4), 544-552. https://doi.org/10.32493/informatika.v5i4.6760.
