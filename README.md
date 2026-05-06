# PoseCertain — Uncertainty-Aware Spacecraft 6D Pose Estimation

> Spacecraft 6D pose estimation with MC Dropout uncertainty quantification for OOD detection on the SPEED dataset.

This project extends [UrsoNet](https://github.com/pedropro/UrsoNet) with **Bayesian uncertainty quantification** via MC Dropout, enabling the model to detect unreliable predictions on out-of-distribution (OOD) real-domain images.

---

## ✨ Key Contributions

- 🔵 **MC Dropout Inference** — 30-sample stochastic forward passes for uncertainty estimation
- 📊 **Orientation Uncertainty** — Predictive Entropy, Expected Entropy (Aleatoric), Mutual Information (Epistemic)
- 📍 **Location Uncertainty** — Gaussian NLL Loss with per-axis epistemic variance output
- 🌍 **OOD Detection** — Evaluated on real SPEED images; epistemic uncertainty correlates with pose error (r=0.73)

---

## 🏗️ Architecture

Built on top of UrsoNet (ResNet50 backbone):
- **Location branch**: Gaussian NLL loss with log-variance output for uncertainty
- **Orientation branch**: Soft classification over 32,768 bins with MC Dropout uncertainty decomposition

---

## 📦 Installation

```bash
git clone https://github.com/你的用户名/PoseCertain.git
cd PoseCertain
pip install -r requirements.txt
```

---

## 📁 Dataset

Download the [SPEED dataset](https://kelvins.esa.int/satellite-pose-estimation-challenge/) and place it under:

```
datasets/
└── speed/
    ├── images/
    └── train.json / test.json
```

---

## 🚀 Pretrained Weights

Download pretrained weights  and place under:(but it is not available)

```
models/logs/speed20260413T1059/weights_best.h5
```

---

## 🔍 Run Uncertainty Evaluation

```bash
# 对真实域图像进行OOD不确定性评估
python pose_estimator.py --dataset speed --mode ood
```

Output: `real_ood_uncertainty.csv` containing per-image uncertainty metrics.

---


## 📜 Citation

This work builds upon UrsoNet:

```bibtex
@article{proenca2019deep,
  title={Deep Learning for Spacecraft Pose Estimation from Photorealistic Rendering},
  author={Proenca, Pedro F and Gao, Yang},
  journal={arXiv preprint arXiv:1907.04298},
  year={2019}
}
```

---

## 📄 License

MIT License
