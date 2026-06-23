# PE-CLIP: A Parameter-Efficient Fine-Tuning of Vision-Language Models for Dynamic Facial Expression Recognition

<p align="center">
  <img src="figures/architecture.png" width="900">
</p>

<p align="center">
  <a href="PAPER_LINK">📄 Paper</a> •
  <a href="#citation">📚 Citation</a>
</p>

Official PyTorch implementation of the ACM TOMM 2026 paper:

**PE-CLIP: A Parameter-Efficient Fine-Tuning of Vision-Language Models for Dynamic Facial Expression Recognition**

---

## 🔥 Highlights

- Parameter-efficient adaptation of CLIP for Dynamic Facial Expression Recognition (DFER).
- Temporal Dynamic Adapter (TDA) for temporal modeling.
- Shared Adapter (ShA) for visual-textual feature refinement.
- Multi-modal Prompt Learning (MaPLe) with AU-guided textual descriptions.
- Less than **6% trainable parameters** while maintaining competitive performance.
- Evaluation on **DFEW**, **FERV39K**, and **AFEW**.
- Includes visualization and explainability tools.

---

## 📖 Abstract

Dynamic Facial Expression Recognition (DFER) remains challenging due to subtle temporal variations, limited training data, and the computational cost of adapting large Vision-Language Models (VLMs). PE-CLIP is a parameter-efficient framework that adapts CLIP for DFER using lightweight adapters and multi-modal prompt learning. The proposed framework introduces a Temporal Dynamic Adapter (TDA), a Shared Adapter (ShA), and AU-guided prompt learning to improve visual-language alignment while requiring only a small fraction of trainable parameters.

---

## 📊 Results

| Dataset | UAR (%) | WAR (%) |
|----------|----------|----------|
| DFEW | 62.82 | 74.04 |
| FERV39K | 41.57 | 51.26 |
| AFEW | 53.85 | 58.49 |

<p align="center">
  <img src="figures/performance_tradeoff.png" width="700">
</p>

---

## 📂 Repository Structure

```text
PE-CLIP
│
├── annotation/           # Dataset annotations
├── dataloader/           # Data loading utilities
├── models/               # PE-CLIP architecture
├── figures/              # README figures
│
├── FERPAmainmxp.py       # Training/testing script (DFEW/FERV39K)
├── FERPAmainmxp_AFEW.py  # Training/testing script (AFEW)
├── fmix.py               # FMix augmentation
├── AttentionMap.py       # Attention map visualization
├── vit_rollout.py        # Attention rollout visualization
├── vit_grad_rollout.py   # Gradient rollout visualization
└── T-SNE Visualization.py # Feature visualization
```

---

## 🛠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- Torchvision
- timm
- einops
- NumPy
- scikit-learn
- matplotlib
- tqdm
- thop

---

## 🚀 Training and Evaluation

### DFEW / FERV39K

```bash
python FERPAmainmxp.py
```

### AFEW

```bash
python FERPAmainmxp_AFEW.py
```

Please update the dataset paths according to your local environment before running the scripts.

---

## 🔍 Visualization

### Attention Maps

```bash
python AttentionMap.py
```

### Attention Rollout

```bash
python vit_rollout.py
```

### Gradient Attention Rollout

```bash
python vit_grad_rollout.py
```

### t-SNE Feature Visualization

```bash
python "T-SNE Visualization.py"
```

---

## 📚 Citation

```bibtex
@article{saadi2026peclip,
  title={PE-CLIP: A Parameter-Efficient Fine-Tuning of Vision-Language Models for Dynamic Facial Expression Recognition},
  author={Saadi, Ibtissam and Hadid, Abdenour and Cunningham, Douglas W. and Taleb-Ahmed, Abdelmalik and El Hillali, Yassin},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2026}
}
```

---

## 👩‍💻 Authors

- Ibtissam Saadi
- Abdenour Hadid
- Douglas W. Cunningham
- Abdelmalik Taleb-Ahmed
- Yassin El Hillali

---

## 🙏 Acknowledgements

This work was conducted within a cotutelle PhD program between BTU Cottbus-Senftenberg, Germany, and Université Polytechnique Hauts-de-France, France.

If you find this repository useful, please consider giving it a ⭐.
