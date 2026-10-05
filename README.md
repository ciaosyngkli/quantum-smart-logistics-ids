# Quantum Smart Logistics IDS (`quantum-smart-logistics-ids`)

This repository is for my cybersecurity research project about using Quantum Machine Learning (QML) to detect network attacks in smart logistics systems. It test how quantum-enhanced classifiers perform compared to standard classical machine learning models using the UNSW-NB15 dataset.

---

## 📌 Project Overview
Smart logistics edge devices need fast and accurate intrusion detection. In this research, I built a hybrid architecture that compress high-dimensional network features down to 4 qubits using an unsupervised Autoencoder, then pass them into a Quantum Support Vector Classifier (QSVC).

### Models Tested in Benchmark
Instead of running just one paper model, the script runs a benchmark comparing 6 different model setups together:

* **Model A:** Classical SVM (All 42 features) - Standard baseline.
* **Model B:** Random Forest (All 42 features) - Upper bound reference.
* **Model C:** QSVC on PCA features - Quantum baseline using PCA reduction.
* **Model D:** Classical SVM on Autoencoder features - Ablation test to check if quantum kernel actually helps.
* **Model E (Proposed Hybrid):** Autoencoder + Quantum Kernel SVM - The proposed QML model.
* **Model F:** Autoencoder-only Anomaly Detector - Unsupervised reconstruction error model.

---
