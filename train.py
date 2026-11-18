"""
=============================================================================
Phase 4: Training Engine, Optimization & Early Stopping by Macro-F1
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Key Optimization Strategies:
-----------------------------------------------------------------------------
1. Optimizer: AdamW with weight decay (0.01) and differential learning rates:
   - 1e-5 for pre-trained backbones (DenseNet-201, ViT-B/16)
   - 1e-4 for newly initialized CBAM attention and Fusion/Classifier Head
2. LR Scheduler: CosineAnnealingLR for smooth convergence to global minima.
3. Loss: Class-Weighted Focal Loss (gamma=2.0) mitigating extreme class skew.
4. Model Selection Criterion: Macro-F1 score with Early Stopping (patience=10).
5. Mixed Precision: Automatic Mixed Precision (AMP) for maximum GPU throughput.
=============================================================================
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from typing import Dict, List, Tuple

from loss import FocalLoss


class EarlyStopping:
    """
    Early stopping handler that monitors Validation Macro-F1 score.
    Saves model checkpoint when Macro-F1 achieves a new peak.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, checkpoint_path: str = "best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = -float('inf')
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """
        Returns True if early stopping should terminate training.
        """
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(self.best_model_weights, self.checkpoint_path)
            print(f"[*] Validation Macro-F1 improved to {current_score:.4f}. Model weights saved to {self.checkpoint_path}")
        else:
            self.counter += 1
            print(f"[!] EarlyStopping counter: {self.counter}/{self.patience} (Best Macro-F1: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def setup_differential_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-2
) -> AdamW:
    """
    Sets up differential learning rates:
    - Backbone features (DenseNet & ViT) receive smaller LR to preserve pretrained representations.
    - CBAM attention & classification head receive larger LR to rapidly learn dermatological fusion.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "densenet_features" in name or "vit" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]

    return AdamW(optimizer_grouped_parameters)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates validation performance."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def fit_model(
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    epochs: int = 50,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    patience: int = 10,
    checkpoint_path: str = "best_model.pth"
) -> Dict[str, List[float]]:
    """
    Full training routine orchestrating AdamW, CosineAnnealingLR, Focal Loss, 
    and Validation Macro-F1 early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Initializing training on target device: {device}")
    model = model.to(device)

    # Multi-Class Focal Loss with computed inverse class weights
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    
    # Differential AdamW Optimizer
    optimizer = setup_differential_optimizer(
        model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=1e-2
    )

    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # AMP Gradient Scaler
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Early Stopping based on Macro-F1
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {
        "train_loss": [], "train_acc": [], "train_macro_f1": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train Macro-F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val Macro-F1:   {val_f1:.4f} | LR: {current_lr:.6f}")

        # Check early stopping condition
        if early_stopping(val_f1, model):
            print(f"\n[Training] Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(f"\n[Training] Finished in {total_time/60:.2f} minutes.")
    print(f"[Training] Peak Validation Macro-F1: {early_stopping.best_score:.4f}")

    # Load best checkpoint weights before returning
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        print("[Training] Best model weights restored to memory.")

    return history
