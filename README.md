# 🧠 SPFT: Optimization-Driven Fine-Tuning for Medical Imaging

A simple and effective training framework for improving **cross-dataset generalization** of foundation models in medical imaging.

This repository provides training scripts demonstrating strong performance across:

* ChestX-ray14 (CXR14)
* CheXpert
* HAM10000

---

## 🚀 Overview

Foundation models achieve strong performance in medical imaging, but generalization across datasets remains challenging.

We propose a **lightweight optimization-driven fine-tuning strategy** based on:

* 🔓 Progressive layer unfreezing
* ⚖️ Class-aware loss weighting
* 📉 Differential learning rates
* 📊 Exponential Moving Average (EMA)

👉 **Key Idea:**
Improve performance through **training strategy**, without modifying model architecture.

---

## 📊 Key Results

* **ChestX-ray14:** ~0.83 Macro AUC
* **CheXpert:** ~0.83–0.85 Macro AUC
* **HAM10000:** ~0.95 Macro AUC

✔ Competitive performance across datasets
✔ Stable training
✔ No architectural modifications

---

## 📂 Repository Structure

```id="7j8f2q"
repo/
│
├── train/
│   ├── train_chexpert.py
│   ├── train_cxr14.py
│   ├── train_ham10000.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash id="t5f3lk"
pip install -r requirements.txt
```

---

## ▶️ Training

### CheXpert

```bash id="6z9vpk"
python train/train_chexpert.py
```

### ChestX-ray14

```bash id="l9u2re"
python train/train_cxr14.py
```

### HAM10000

```bash id="b3kx2c"
python train/train_ham10000.py
```

---

## 🧪 Method

The training pipeline consists of:

1. Train classification head (frozen encoder)
2. Progressive unfreezing of top layers
3. Apply class-aware loss weighting
4. Stabilize training using EMA

---

## 🔬 Reproducibility

Results reported in the paper can be reproduced using the provided training scripts.

Pretrained model weights will be released upon acceptance.

---

## 📄 Citation

```id="cz7w2v"
@article{spft2026,
  title={Optimization-Driven Fine-Tuning for Cross-Dataset Generalization in Medical Imaging},
  author={Anonymous},
  year={2026}
}
```

---

## ⭐ Contribution

This work shows that:

> Strong performance in medical imaging can be achieved through **optimization strategies rather than architectural changes**.
