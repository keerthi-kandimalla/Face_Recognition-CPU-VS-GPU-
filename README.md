# CPU vs GPU Acceleration of CNN-Based Face Recognition
## A Complete Computer Architecture Performance Analysis Website

**Academic Project Title:** *CPU vs GPU Acceleration of CNN-Based Telugu Celebrity Face Recognition: A Computer Architecture Performance Analysis*

---

### 1. Executive Summary

This research engineering project combines:
- **Computer Organization & Architecture (COA)**
- **Deep Learning & Convolutional Neural Networks (CNN)**
- **Computer Vision & Face Biometrics**
- **Hardware Acceleration & Parallel Processing**

The system investigates how underlying processor architectures impact real-world deep neural network workloads. Specifically, it contrasts **latency-optimized sequential multicore CPUs** (Intel Core i5-13500H) against **throughput-optimized massively parallel vector Execution Units (EUs)** on integrated GPUs (Intel Iris Xe Graphics) using **Intel OpenVINO** and **DirectX 12 DirectML**.

---

### 2. Physical Laptop Hardware Profile

- **Processor (CPU):** 13th Gen Intel(R) Core(TM) i5-13500H (12 Cores: 4 P-Cores + 8 E-Cores, 16 Logical Threads)
- **Integrated Graphics (GPU):** Intel(R) Iris(R) Xe Graphics (iGPU), 80-96 Vector Execution Units
- **Driver Version:** 31.0.101.4032 (WDDM 3.1)
- **Dedicated / Unified Memory:** 1024 MB Dynamic Unified Shared Memory (USM)
- **Host Operating System:** Microsoft Windows 11 Home Single Language (Build 10.0.26200)
- **Runtime Environment:** Python 3.13.3 (AMD64)
- **Acceleration Frameworks:** Intel OpenVINO 2026.4.1 (Native CPU & Intel GPU plugins) + ONNX Runtime DirectML 1.24.4 (DirectX 12)

---

### 3. Core Architecture & Modules

```
face__recognition/
├── dataset/                    # Telugu Celebrity Dataset Folders
│   ├── Vijay_Deverakonda/
│   ├── Allu_Arjun/
│   ├── Jr_NTR/
│   ├── Ram_Charan/
│   ├── Mahesh_Babu/
│   ├── Prabhas/
│   └── ... (18 Telugu Celebrities)
├── models/                     # Pretrained CNN Weights & Models
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── database/                   # SQLite Persistent Storage
│   └── face_recognition.db
├── backend/                    # Python Backend Pipeline
│   ├── main.py                 # FastAPI REST Endpoints & Web Server
│   ├── hardware.py             # Hardware Detection & Live Metrics
│   ├── face_engine.py          # YuNet Detector & ArcFace/SFace Extractor
│   ├── benchmark.py            # Comparative CPU vs GPU Benchmarking Suite
│   ├── database.py             # SQLite ORM & Query Helpers
│   ├── dataset_service.py      # Dataset Importer, ZIP Parser & Validator
│   └── populate_dataset.py     # Initial Telugu Dataset Initializer
├── frontend/                   # Modern Web Application Interface
│   ├── index.html              # Multi-Tab Dashboard & COA Visualizers
│   ├── app.js                  # Frontend Controllers & Chart.js Hooks
│   └── styles.css              # Custom Animations & Themes
├── tests/
│   └── test_pipeline.py        # Automated 7-Step Verification Suite
├── requirements.txt            # Python Dependencies
├── run_server.py               # Application Launch Script
└── README.md                   # Academic Documentation
```

---

### 4. Non-Negotiable Features Implemented

#### A. Telugu Celebrity Face Recognition System
- **18 Prominent Celebrities Registered:**
  1. Vijay Deverakonda
  2. Allu Arjun
  3. Jr. NTR
  4. Ram Charan
  5. Mahesh Babu
  6. Prabhas
  7. Pawan Kalyan
  8. Nani
  9. Chiranjeevi
  10. Ravi Teja
  11. Ram Pothineni
  12. Naga Chaitanya
  13. Rashmika Mandanna
  14. Samantha Ruth Prabhu
  15. Sai Pallavi
  16. Anushka Shetty
  17. Pooja Hegde
  18. Keerthy Suresh
- **Detection Pipeline:** OpenCV YuNet CNN detector with 5 facial landmark alignment.
- **Biometric Matching:** Cosine Similarity comparison over L2-normalized 512-D embeddings with calibrated threshold. Returns recognized celebrity name or explicit **`UNKNOWN PERSON`** status.

#### B. Compulsory CPU vs GPU Performance Benchmarking Dashboard
- **Experiment A (Single Image):** Side-by-side pure inference comparison. Demonstrates why CPU single-thread frequency and cache locality are competitive at batch size 1 due to GPU kernel launch and bus overhead.
- **Experiment B (Large-Scale CNN Benchmark):** Tests workload sizes (10, 50, 100, 500, 1000, 2000 images) and batch sizes (1, 8, 16, 32, 64). Demonstrates GPU parallel compute scaling.
- **Empirical Measured Metrics:**
  - Speedup: $\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{GPU}}}$
  - Throughput: $\text{FPS} = \frac{N_{\text{images}}}{T_{\text{total}}}$
  - Percentage Improvement: $\left(\frac{T_{\text{CPU}} - T_{\text{GPU}}}{T_{\text{CPU}}}\right) \times 100\%$

---

### 5. Verified Experimental Results (On User's Dell Hardware)

| Workload Configuration | CPU (Intel Core i5-13500H) | GPU (Intel Iris Xe Graphics) | Speedup Factor | Faster Device |
|:---|:---:|:---:|:---:|:---:|
| **Batch Size 1 (Single Face)** | ~15.9 ms | ~18.5 ms | **0.86x** | **CPU** (Low dispatch overhead) |
| **Batch Size 8 (16 Images)** | 282.26 ms (56.7 FPS) | 71.87 ms (222.5 FPS) | **3.93x** | **GPU** (Vector SIMD EUs) |
| **Batch Size 16 (50 Images)** | 884.10 ms (56.5 FPS) | 215.30 ms (232.2 FPS) | **4.10x** | **GPU** (High Parallel Throughput) |

---

### 6. How to Run the Website

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the automated test suite
python -m tests.test_pipeline

# 3. Launch the web application
python run_server.py
```

Then open your web browser at:
**`http://localhost:8000`**
