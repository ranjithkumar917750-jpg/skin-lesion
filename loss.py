"""
=============================================================================
Phase 3: Handling Class Imbalance - Multi-Class Focal Loss
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Mathematical Formulation:
-----------------------------------------------------------------------------
Standard Cross-Entropy treats all classes and error magnitudes equally:
    CE(p_t) = -log(p_t)

In HAM10000, ~67% of cases are Melanocytic Nevi ('nv'). Easy negative examples 
dominate the gradient, suppressing learning on critical malignant cases 
(e.g., Melanoma 'mel', Basal Cell Carcinoma 'bcc', Actinic Keratosis 'akiec').

Focal Loss introduces:
1. Modulating Factor: (1 - p_t)^gamma
   - When an easy sample has high confidence (p_t -> 1), (1 - p_t)^gamma -> 0,
     diminishing its loss contribution.
   - When a difficult sample has low confidence (p_t -> 0), (1 - p_t)^gamma -> 1,
     preserving full gradient magnitude.
2. Alpha Weighting: alpha_t
   - Inversely balances class frequencies across the 7 dermatological categories.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with class-weighting and numerical stabilization.
    """
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha (Tensor, list, or float, optional): Class weights alpha_t. If tensor or list, 
                                                    must match number of classes.
            gamma (float): Focusing parameter gamma >= 0 (default: 2.0).
            reduction (str): 'mean', 'sum', or 'none'.
            label_smoothing (float): Label smoothing coefficient in [0, 1].
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor([alpha], dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.
        
        Args:
            inputs (torch.Tensor): Unnormalized predicted logits of shape (B, C).
            targets (torch.Tensor): Ground-truth class indices of shape (B,).
            
        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Compute log probabilities and probabilities with numerical stability
        log_p = F.log_softmax(inputs, dim=1)  # (B, C)
        p = torch.exp(log_p)                  # (B, C)

        # Gather probability corresponding to target class
        targets = targets.view(-1, 1)         # (B, 1)
        log_pt = log_p.gather(1, targets)     # (B, 1)
        pt = p.gather(1, targets)             # (B, 1)

        # Compute focal weighting factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Apply class balancing alpha if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.squeeze(1)).view(-1, 1)
            loss = -at * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        # Apply label smoothing regularization if enabled
        if self.label_smoothing > 0.0:
            smoothed_loss = -log_p.mean(dim=1, keepdim=True)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smoothed_loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.squeeze(1)


def compute_effective_num_class_weights(class_counts: list, beta: float = 0.9999) -> torch.Tensor:
    """
    Computes class weights based on the 'Class-Balanced Loss Based on Effective Number of Samples'
    (Cui et al., CVPR 2019):
        E_n = (1 - beta^n) / (1 - beta)
        weight_c = 1 / E_n_c
    
    Args:
        class_counts (list or np.ndarray): Number of samples per class.
        beta (float): Hyperparameter in [0, 1). Default 0.9999.
        
    Returns:
        torch.Tensor: Normalized weight tensor of shape (num_classes,).
    """
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
    # Normalize weights so sum equals num_classes
    weights = weights / weights.sum() * len(class_counts)
    return weights
