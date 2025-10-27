# 🫁 Semantic Segmentation of COVID-19 CT Scans

**Deep Learning for automatic segmentation of COVID-19 lung infections from CT images using a U-Net model.**  
Developed and maintained by [**Martin Badrous**](https://github.com/martinbadrous).

---

## 📘 Overview

This project implements a **semantic segmentation pipeline** for identifying COVID-19 infection regions in lung CT scans.  
Using a **U-Net architecture** trained with a **BCE + Dice loss**, the model learns to segment infected lung areas from grayscale CT slices.

The goal of this project is to provide an open and reproducible baseline for medical image segmentation tasks, enabling further research on **infection quantification**, **disease progression**, and **clinical decision support**.

---

## 🧱 Repository Structure

```bash
Semantic-Segmentation-COVID-19/
├── README.md
├── requirements.txt
│
├── data/
│   ├── dataset.py          # Data loading, preprocessing, and augmentation
│   └── split_data.py       # Optional: helper for creating train/val splits
│
├── models/
│   └── unet.py             # U-Net model
│
├── utils/
│   ├── losses.py           # Dice, BCE, BCE+Dice loss functions
│   ├── metrics.py          # Dice / IoU metrics
│   └── helpers.py          # Mask saving and visualization helpers
│
├── train.py                # Training script (with validation + checkpoint)
├── infer.py                # Inference script (predict mask from CT image)
└── checkpoints/             # Saved models during training
```

---

## 🧬 Dataset Preparation

This project supports any CT dataset containing **image-mask pairs**.

### ✅ Recommended Datasets
- [**MedSeg COVID-19 CT Lung and Infection Segmentation Dataset**](https://medicalsegmentation.com/covid19/)
- [**COVID-19 CT Lung and Infection Segmentation Dataset (Ma et al., 2020)**](https://zenodo.org/record/3757476)

### 📂 Expected Directory Layout
```bash
data/COVID/
├── train_images/
│   ├── case001_slice01.png
│   ├── case001_slice02.png
│   └── ...
├── train_masks/
│   ├── case001_slice01_mask.png
│   ├── case001_slice02_mask.png
│   └── ...
├── val_images/
│   └── ...
└── val_masks/
    └── ...
```

Mask filenames must match image names (optionally with `_mask` suffix).  
You can use `split_data.py` to create train/val splits automatically.

---

## ⚙️ Environment Setup

Install dependencies via pip or conda.

```bash
conda create -n covidseg python=3.9
conda activate covidseg

pip install -r requirements.txt
```

---

## 🧠 Training the Model

Edit the dataset paths in `train.py`, then run:

```bash
python train.py
```

The script will:
- Train U-Net on the COVID CT dataset
- Validate each epoch
- Log **loss**, **Dice**, and **IoU**
- Save the best checkpoint automatically under `checkpoints/best_model.pth`

**Example console output:**

```
Epoch 15/50
train_loss: 0.2410 | val_loss: 0.1923 | val_dice: 0.863 | val_iou: 0.781
✔ Saved best model: checkpoints/best_model.pth
```

---

## 🔍 Inference / Prediction

Predict lesion masks for new CT images using a trained model.

```bash
python infer.py
```

Example output files:
```
demo_image.png        → Input CT slice
prediction_mask.png   → Predicted lesion segmentation
```

You can visualize the output in any image viewer (white = predicted infection).

---

## 📈 Evaluation Metrics

| Metric | Description | Formula |
|--------|--------------|----------|
| **Dice Coefficient** | Measures overlap between prediction and ground truth | 2TP / (2TP + FP + FN) |
| **IoU (Jaccard)** | Intersection over Union | TP / (TP + FP + FN) |
| **BCE Loss** | Binary Cross Entropy | -ylog(p) - (1-y)log(1-p) |

---

## 🧩 Model Architecture

The U-Net consists of an encoder-decoder structure with skip connections to retain spatial features.

```
Input → [Conv-BN-ReLU]x2 → Down → ... → Bottleneck → Up → [Conv-BN-ReLU]x2 → Sigmoid → Output Mask
```

- Input size: (1 x 256 x 256)
- Output: Binary mask (1 x 256 x 256)
- Parameters: ~7M

---

## 🧪 Example Results

| Metric | Score (Example Run) |
|--------|---------------------|
| Dice Coefficient | **0.86 ± 0.02** |
| IoU              | **0.79 ± 0.03** |

Visual results show strong lesion localization with smooth, connected boundaries.

---

## 🧠 Applications

- COVID-19 lesion quantification  
- Medical dataset preprocessing  
- Baseline for 2D segmentation tasks (lungs, organs, tumors)  
- Clinical visualization tools  

---

## 🧾 Citations & Acknowledgments

This work is inspired by open COVID-19 datasets and research in medical image segmentation.  
If you use this repository, please cite the following datasets:

- **Ma et al., 2020**, *COVID-19 CT Lung and Infection Segmentation Dataset*, Zenodo.  
- **MedSeg COVID-19 CT Segmentation Dataset**, Medical Segmentation Decathlon, 2020.

---

## 👨‍💻 Author

**Martin Badrous**  
Computer Vision & Robotics Engineer  
🎓 M.Sc. in Computer Vision & Robotics, Université de Bourgogne  
📧 martin.badrous@gmail.com  
🔗 [GitHub Profile](https://github.com/martinbadrous)

---

## 🪪 License

MIT License © 2025 Martin Badrous  
Feel free to use, modify, and share this project for academic or research purposes.
