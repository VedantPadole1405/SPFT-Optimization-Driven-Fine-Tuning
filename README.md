# 🧠 SPFT: Optimization-Driven Fine-Tuning for Medical Imaging

A simple yet effective training framework for improving **cross-dataset generalization** of foundation models in medical imaging.

This repository implements our optimization-driven fine-tuning strategy, demonstrating strong performance across:

* ChestX-ray14 (CXR14)
* CheXpert
* HAM10000

---

## 🚀 Overview

Foundation models have shown strong performance in medical imaging, but generalization across datasets and modalities remains challenging.

We propose a **lightweight adaptation strategy** based on:

* 🔓 Progressive layer unfreezing
* ⚖️ Class-aware loss weighting
* 📉 Differential learning rates
* 📊 Exponential Moving Average (EMA) stabilization

👉 **Key Idea:**
Instead of modifying architectures, we improve performance through **carefully designed optimization strategies**.

---

## 📊 Key Results

* **ChestX-ray14:** ~0.83 Macro AUC
* **CheXpert:** ~0.83–0.85 Macro AUC
* **HAM10000:** ~0.95 Macro AUC

✔ Competitive performance across datasets
✔ Stable training with low variance
✔ No architectural modifications required

---

## 📂 Repository Structure

```
repo/
│
├── train/
│   ├── train_chexpert.py
│   ├── train_cxr14.py
│   ├── train_ham10000.py
│
├── models/
│   ├── model.py
│
├── utils/
│   ├── dataset.py
│   ├── evaluation.py
│
├── checkpoints/   # optional
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

### Train on CheXpert

```bash
python train/train_chexpert.py
```

### Train on ChestX-ray14

```bash
python train/train_cxr14.py
```

### Train on HAM10000

```bash
python train/train_ham10000.py
```

---

## 🧪 Method Details

Our training framework (SPFT) consists of:

* **Stage 1:** Train classification head (frozen encoder)
* **Stage 2:** Progressive unfreezing of top layers
* **Stage 3:** Apply class-aware loss weighting
* **Stage 4:** Stabilize training using EMA

This approach enables:

* Better adaptation to dataset-specific distributions
* Improved performance on imbalanced classes
* Stable convergence across runs

---

## 🔬 Reproducibility

Results reported in the paper can be reproduced using the provided training scripts.

---

## 📌 Notes

* Designed for **research reproducibility**, not production deployment
* Dataset preprocessing may vary depending on source
* Performance may vary slightly across runs

---

## 📄 Citation

If you find this work useful, please consider citing:

```
@article{spft2026,
  title={Optimization-Driven Fine-Tuning for Cross-Dataset Generalization in Medical Imaging},
  author={Anonymous},
  year={2026}
}
```

---

## 🤝 Acknowledgements

Built using:

* PyTorch
* HuggingFace Transformers
* RAD-DINO backbone

---

## ⭐ Contribution

This work demonstrates that:

> Strong performance in medical imaging can be achieved through **optimization strategies rather than architectural changes**.

---

## 📬 Contact

For questions or issues, feel free to open an issue in this repository.
