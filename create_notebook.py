"""
Generates a 100% self-contained, standalone Jupyter/Colab notebook.
Zero external .py file dependencies - imports and definitions are strictly ordered
so any user can open it in Google Colab and run with 1-click on a free GPU.
"""
import json
import re

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "name": "SOTA_Skin_Lesion_Classification_Hybrid_3Brain.ipynb",
            "provenance": [],
            "accelerator": "GPU"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Title
add_md("""# 🔬 SOTA Skin Lesion Classification ('3-Brain' Hybrid Architecture)
### Target: 96-97% Diagnostic Accuracy on 7-Class HAM10000 Dataset
**DenseNet-201 (Local) + CBAM (Attention) + ViT-B/16 (Global Self-Attention) + Multiscale Fusion**
""")

# Cell 1: Environment Setup & Installs
add_md("""---
## 📦 Block 1: Environment Setup & Hardware Verification
Installs required packages and verifies NVIDIA GPU acceleration (T4 / A100 / V100).
""")

cell_1 = """# 1. Install dependencies
!pip install -q albumentations opencv-python-headless scikit-learn seaborn kagglehub tqdm

import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from typing import Optional, Tuple, Dict, List

# Hardware verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Active PyTorch Device: {device}")
if torch.cuda.is_available():
    print(f"[Hardware] GPU Accelerator: {torch.cuda.get_device_name(0)}")
    print(f"[Hardware] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[WARNING] Running on CPU. For fast training, enable GPU via Runtime -> Change runtime type -> T4 GPU.")
"""
add_code(cell_1)

# Cell 2: Preprocessing (DullRazor + CLAHE)
add_md("""---
## 🧪 Block 2: Clinical Preprocessing (DullRazor + CLAHE in CIE-LAB)
1. **DullRazor Morphological Inpainting:** Isolates hair artifacts via Black-Hat morphological filtering and restores skin textures with Telea inpainting.
2. **CIE-LAB CLAHE Equalization:** Balances lighting variations strictly on the $L$ (Luminance) channel while preserving clinical chrominance ($A, B$).
""")

with open("preprocessing.py", "r", encoding="utf-8") as f:
    prep_code = f.read()

# Strip any main execution or standalone tests
prep_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', prep_code)
add_code(prep_clean.strip())

# Cell 3: Dataset Ingestion & Lesion-Stratified Splitting
add_md("""---
## 📂 Block 3: HAM10000 Automated Dataset Ingestion & Zero-Leakage Split
- Automatically downloads the HAM10000 dataset via `kagglehub`.
- Performs **Lesion-ID Stratified Partitioning** guaranteeing 0% data leakage across patients.
- Creates class-balanced DataLoaders with Albumentations data augmentation.
""")

with open("dataset.py", "r", encoding="utf-8") as f:
    dataset_code = f.read()

# Strip local import of preprocessing since it's already defined
dataset_clean = re.sub(r'from preprocessing import [^\n]+\n', '', dataset_code)
dataset_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', dataset_clean)
add_code(dataset_clean.strip())

# Cell 4: Model Architecture (3-Brain Hybrid)
add_md("""---
## 🧠 Block 4: The '3-Brain' Hybrid Deep Learning Architecture
1. **Brain 1 (Local):** DenseNet-201 capturing dense micro-textures (1920-d).
2. **Brain 2 (Attention):** CBAM (Channel & Spatial Attention) focusing on lesions.
3. **Brain 3 (Global Context):** Vision Transformer (ViT-B/16) modeling long-range non-local patch dependencies via [CLS] token (768-d).
4. **Multiscale Fusion:** $1920 + 768 = 2688$-d fused representation with deep regularized MLP head.
""")

with open("model.py", "r", encoding="utf-8") as f:
    model_code = f.read()
model_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', model_code)
add_code(model_clean.strip())

# Cell 5: Class-Weighted Multi-Class Focal Loss & Training Engine
add_md(r"""---
## ⚖️ Block 5: Multi-Class Focal Loss & Optimization Engine
- **Class-Weighted Focal Loss ($\gamma = 2.0$):** Suppresses majority class gradients (nevi) and boosts learning on under-represented malignancies (melanoma, dermatofibroma).
- **AdamW + Differential Learning Rates:** $1e-5$ for backbones, $1e-4$ for CBAM attention and fusion head.
- **Cosine Annealing & Early Stopping:** Monitors validation Macro-F1 score to prevent overfitting.
""")

with open("loss.py", "r", encoding="utf-8") as f:
    loss_code = f.read()
with open("train.py", "r", encoding="utf-8") as f:
    train_code = f.read()

# Strip local import
train_clean = re.sub(r'from loss import [^\n]+\n', '', train_code)
train_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', train_clean)
loss_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', loss_code)

add_code(loss_clean.strip() + "\n\n" + train_clean.strip())

# Cell 6: Evaluation (TTA + MC Dropout) & Explainable AI (Grad-CAM)
add_md("""---
## 📊 Block 6: Evaluation (5-Fold TTA + MC Dropout) & Grad-CAM Explainability
- **5-Fold Test-Time Augmentation (TTA):** Eliminates boundary variance and maximizes diagnostic accuracy.
- **Monte Carlo Dropout Uncertainty:** Samples 10 stochastic forward passes to quantify epistemic uncertainty.
- **Grad-CAM:** Generates high-resolution heatmaps overlaid on dermoscopy images.
""")

with open("evaluate.py", "r", encoding="utf-8") as f:
    eval_code = f.read()
with open("gradcam.py", "r", encoding="utf-8") as f:
    gradcam_code = f.read()

eval_clean = re.sub(r'from dataset import [^\n]+\n', '', eval_code)
eval_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', eval_clean)

gradcam_clean = re.sub(r'from dataset import [^\n]+\n', '', gradcam_code)
gradcam_clean = re.sub(r'if __name__ == "__main__":[\s\S]*$', '', gradcam_clean)

add_code(eval_clean.strip() + "\n\n" + gradcam_clean.strip())

# Cell 7: Full Execution Loop & Real Weights Download
add_md("""---
## 🚀 Block 7: Complete End-to-End Real Training & Evaluation
Executes the full pipeline on real HAM10000 images, evaluates on unseen test lesions, and exports `best_model.pth`.
""")

execution_code = """# ===========================================================================
# Full End-to-End Execution on Genuine HAM10000 Dataset
# ===========================================================================
DATA_DIR = "./data/ham10000"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Download full dataset
print("[*] Downloading / Resolving HAM10000 Dataset...")
dataset_path = download_and_extract_ham10000(DATA_DIR)

# 2. Build metadata DataFrame with zero-leakage patient-level partition
df = prepare_ham10000_dataframe(dataset_path)
print(f"Total dataset indexed: {len(df)} images across {df['lesion_id'].nunique()} unique lesions.")

train_df, val_df, test_df = create_stratified_splits(df, val_ratio=0.15, test_ratio=0.15, random_state=42)
print(f"Partition: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} images.")

# 3. Create DataLoaders
train_loader, val_loader, test_loader, class_weights = build_dataloaders(
    train_df, val_df, test_df,
    batch_size=32 if torch.cuda.is_available() else 16,
    num_workers=2 if torch.cuda.is_available() else 0,
    img_size=224,
    use_weighted_sampler=True
)

# 4. Instantiate 3-Brain Model
print("[*] Instantiating 3-Brain Hybrid Model...")
model = build_model(num_classes=7, pretrained=True, dropout_rate=0.4).to(device)

# 5. Train with Focal Loss and Cosine Annealing (Monitored by Macro-F1)
print("[*] Beginning Genuine Training across HAM10000...")
history = fit_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    class_weights=class_weights,
    epochs=40,
    backbone_lr=1e-5,
    head_lr=1e-4,
    patience=8,
    checkpoint_path="best_model.pth"
)

# 6. Evaluate on Held-Out Test Partition with 5-Fold TTA & Monte Carlo Dropout
print("\\n" + "="*80)
print(" FINAL HELD-OUT TEST EVALUATION (5-Fold TTA + MC Dropout)")
print("="*80)
results = evaluate_test_set(
    model=model,
    test_loader=test_loader,
    device=device,
    use_tta=True,
    use_mc_dropout=True,
    mc_passes=10
)

# Plot & Save Confusion Matrix
plot_confusion_matrix(results["confusion_matrix"], save_path="confusion_matrix.png")
print(f"\\n[+] Real Test Accuracy: {results['accuracy']*100:.2f}%")
print(f"[+] Real Macro F1-Score: {results['macro_f1']:.4f}")

# 7. Generate Grad-CAM Diagnostic Visualization
print("\\n[*] Generating Grad-CAM Diagnostic Visualization on Test Lesion...")
sample_images, sample_targets, sample_ids = next(iter(test_loader))
sample_img = sample_images[0:1].to(device)

gradcam_engine = GradCAM(model)
heatmap, pred_class, confidence = gradcam_engine.generate_heatmap(sample_img)

# Unnormalize for display
img_vis = sample_img[0].cpu().numpy().transpose(1, 2, 0)
img_vis = (img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
img_vis = np.clip(img_vis * 255.0, 0, 255).astype(np.uint8)

visualize_gradcam_diagnosis(
    image_preprocessed=img_vis,
    cam=heatmap,
    predicted_class=pred_class,
    confidence=confidence,
    ground_truth=sample_targets[0].item(),
    save_path="gradcam_diagnosis.png"
)

# 8. Download Checkpoint to Local Machine (Colab Helper)
try:
    from google.colab import files
    print("\\n[+] Triggering browser download of genuine 'best_model.pth' and 'confusion_matrix.png'...")
    files.download('best_model.pth')
    files.download('confusion_matrix.png')
except Exception as e:
    print(f"Run outside Colab or files.download skipped: {e}")
"""
add_code(execution_code)

output_path = "skin_lesion_classification_colab.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully regenerated: {output_path} with {len(notebook['cells'])} self-contained cells!")
