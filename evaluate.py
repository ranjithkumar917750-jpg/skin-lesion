"""
=============================================================================
Phase 4 & 5: Inference, Test-Time Augmentation (TTA), Monte Carlo Uncertainty
             and Comprehensive Evaluation
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Techniques Implemented:
-----------------------------------------------------------------------------
1. Test-Time Augmentation (TTA):
   - 5-fold spatial transformations (Original, H-Flip, V-Flip, Rot90, Rot270)
   - Averages predictive probabilities to smooth decision boundaries and 
     boost final evaluation accuracy.
2. Monte Carlo (MC) Dropout:
   - Activates dropout modules during inference across 10 stochastic passes.
   - Quantifies prediction variance & epistemic uncertainty for clinical safety.
3. Diagnostic Metrics Suite:
   - Accuracy, Macro-F1, Weighted-F1, Sensitivity (Recall), Specificity.
   - Multi-class Confusion Matrix with normalized counts.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from typing import Dict, Tuple, List
from dataset import LABEL_TO_IDX, IDX_TO_LABEL, CLASS_FULL_NAMES


def enable_mc_dropout(model: nn.Module):
    """
    Sets Dropout layers to train mode while keeping BatchNorm/LayerNorm in eval mode.
    This enables Monte Carlo sampling of model parameter distributions.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_with_tta(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Applies Test-Time Augmentation (TTA) with 5 spatial variants.
    
    Variants:
    1. Original: x
    2. Horizontal Flip: torch.flip(x, dims=[3])
    3. Vertical Flip: torch.flip(x, dims=[2])
    4. 90-degree Rotation: torch.rot90(x, 1, dims=[2, 3])
    5. 270-degree Rotation: torch.rot90(x, 3, dims=[2, 3])
    
    Args:
        model: Trained PyTorch model.
        image_tensor: Batch of images [B, 3, 224, 224].
        device: Device to run computation on.
        
    Returns:
        torch.Tensor: Averaged Softmax probability tensor [B, num_classes].
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    variants = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),
        torch.flip(image_tensor, dims=[2]),
        torch.rot90(image_tensor, 1, dims=[2, 3]),
        torch.rot90(image_tensor, 3, dims=[2, 3])
    ]
    
    probs_list = []
    for var in variants:
        logits = model(var)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs)
        
    # Average predicted probabilities across 5 variants
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs


@torch.no_grad()
def predict_with_mc_dropout(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    num_passes: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs Monte Carlo (MC) Dropout across `num_passes` stochastic forward passes.
    Strictly restores deterministic model.eval() upon completion.
    
    Args:
        model: Trained model.
        image_tensor: Batch of images [B, 3, 224, 224].
        device: Target device.
        num_passes: Number of stochastic passes (default: 10).
        
    Returns:
        tuple: (mean_probabilities [B, C], prediction_variance [B, C], uncertainty_score [B])
    """
    try:
        enable_mc_dropout(model)
        image_tensor = image_tensor.to(device)
        
        stochastic_probs = []
        with torch.no_grad():
            for _ in range(num_passes):
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1)
                stochastic_probs.append(probs)
            
        # Shape: [num_passes, B, C]
        stochastic_probs = torch.stack(stochastic_probs, dim=0)
        
        # 1. Mean probability: E[p]
        mean_probs = stochastic_probs.mean(dim=0)
        
        # 2. Predictive Variance: Var(p) = E[(p - E[p])^2]
        pred_variance = stochastic_probs.var(dim=0)
        
        # 3. Scalar Epistemic Uncertainty per sample: Mean variance across classes
        uncertainty_score = pred_variance.mean(dim=1)
        
        return mean_probs, pred_variance, uncertainty_score
    finally:
        # Strictly restore model to deterministic evaluation mode
        model.eval()


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader,
    device: torch.device,
    use_tta: bool = True,
    use_mc_dropout: bool = True,
    mc_passes: int = 10
) -> Dict[str, any]:
    """
    Evaluates model on the unseen test dataset with TTA and Monte Carlo Uncertainty Estimation.
    Strictly enforces model.eval() and torch.no_grad() throughout evaluation.
    """
    print(f"\n[Evaluation] Evaluating Test Set (TTA: {use_tta}, MC-Dropout Passes: {mc_passes if use_mc_dropout else 0})...")
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0]
            targets = batch[1]
            targets_np = targets.numpy()
            all_targets.extend(targets_np)

            # Enforce deterministic eval mode before computing main prediction probabilities
            model.eval()

            if use_tta:
                # 5-fold Test-Time Augmentation
                probs = predict_with_tta(model, images, device)
            else:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)

            if use_mc_dropout:
                # Compute epistemic uncertainty via MC-Dropout
                _, _, uncertainties = predict_with_mc_dropout(
                    model, images, device, num_passes=mc_passes
                )
                all_uncertainties.extend(uncertainties.cpu().numpy())
                # Restore deterministic eval mode immediately
                model.eval()

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    if len(all_uncertainties) > 0:
        all_uncertainties = np.array(all_uncertainties)

    # Compute Global Performance Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    # Per-Class Sensitivity (Recall) and Specificity
    target_names = [IDX_TO_LABEL[i] for i in range(len(IDX_TO_LABEL))]
    cm = confusion_matrix(all_targets, all_preds)
    
    class_metrics = {}
    for i, name in enumerate(target_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        class_metrics[name] = {
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Support": int(tp + fn)
        }

    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
        "uncertainties": all_uncertainties
    }

    # Print Clinical Diagnostic Summary
    print("\n" + "="*75)
    print("                SOTA SKIN LESION TEST EVALUATION REPORT")
    print("="*75)
    print(f"Overall Accuracy:  {accuracy*100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    if len(all_uncertainties) > 0:
        print(f"Average Epistemic Uncertainty (Variance): {np.mean(all_uncertainties):.6f}")
    print("-"*75)
    print(f"{'Class':<8} | {'Sensitivity':<12} | {'Specificity':<12} | {'Precision':<10} | {'Support':<8} | {'Condition'}")
    print("-"*75)
    for name in target_names:
        m = class_metrics[name]
        full_name = CLASS_FULL_NAMES.get(name, name)
        print(f"{name:<8} | {m['Sensitivity (Recall)']*100:>10.2f}% | {m['Specificity']*100:>10.2f}% | {m['Precision']*100:>8.2f}% | {m['Support']:<8} | {full_name}")
    print("="*75)
    
    return results


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    """
    Renders and saves a normalized confusion matrix heatmap.
    """
    target_names = [IDX_TO_LABEL[i] for i in range(len(IDX_TO_LABEL))]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title("Normalized Confusion Matrix - '3-Brain' Hybrid Architecture", fontsize=14, pad=15)
    plt.xlabel("Predicted Diagnostic Class", fontsize=12)
    plt.ylabel("Ground Truth Lesion Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix saved to: {save_path}")


if __name__ == "__main__":
    from generate_ieee_full_benchmark import generate_full_ham10000_benchmark
    print("\n[+] Executing Official HAM10000 Test Set Evaluation Suite...")
    generate_full_ham10000_benchmark()

