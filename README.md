# Perbandingan Performa antara Plain-34 dan ResNet-34 (Deep Learning RA)

## **Anggota Kelompok**

| **Nama**                    | **NIM**   |
| --------------------------- | --------- | 
| Alfajar                 | 122140122 |
| Ikhsannudin Lathief     | 122140137 |
| Shintya Ayu Wardani     | 122140138 | 
---

Berikut link source code : <a href="https://colab.research.google.com/drive/17uu26xLrM-hf8yF04S_icx59O8YLP-5k?usp=sharing">Collab

## **Perbandingan Metrik**

Berikut merupakan tabel perbandingan antara Plain-34 dan ResNet-34

### Tahap 1

| **No.** | **Perbandingan Metrik**         | **Plain-34**  | **ResNet-34** |
|-----|-------------------------------------|---------------|----------------|
| 1   | `training accuracy`                 |     28.7%     |   50%          |
| 2   | `validation accuracy`               |     20%       |   27%          |
| 3   | `training loss`                     |     1.59      |   1.18         |
| 4   | `validation loss`                   |     8         |   8            |
---

### Tahap 2

| **No.** | **Perbandingan Metrik**         | **Plain-34**  | **ResNet-34** |
|-----|-------------------------------------|---------------|----------------|
| 1   | `training accuracy`                 |     46.6%     |   47.7%        |
| 2   | `validation accuracy`               |     30%       |   32%          |
| 3   | `training loss`                     |     1.30      |   1.27         |
| 4   | `validation loss`                   |     5         |   2            |
---

## **Grafik Sederhana**

![Tahap 1](Tahap%201.jpg)

![Tahap 2](Tahap%202.jpg)

## **Analisis**

Bisa dilihat dari grafik tahap 1 dan tahap 2 bahwa, ResNet-34 terbukti lebih unggul dari Plain-34 karena menghasilkan akurasi lebih tinggi dan loss lebih rendah. hal ini dikarenakan residual connection yang mempermudah aliran gradien sehingga dapat mengatasi masalah vanishing gradient, serta membuat proses training lebih stabil yang membuat model lebih cepat konvergen dan mampu melakukan generalisasi dengan lebih baik.

## **Konfigurasi Hyperparameter**

| **No.** | **Hyperparameter** | **Plain-34**      | **ResNest-34**   |
|-----|------------------------|-------------------|------------------|
| 1   | `epoch`                |        2          |        2         |
| 2   | `optimizer`            |      AdamW        |      AdamW       |
| 3   | `loss function`        | Cross-entropy loss|Cross-entropy loss|
| 4   | `learning rate`        |       0.001       |      0.001       |
---
