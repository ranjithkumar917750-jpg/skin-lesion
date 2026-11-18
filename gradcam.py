"""
=============================================================================
Phase 5: Explainable AI (XAI) - Grad-CAM Implementation for Hybrid Model
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Gradient-Weighted Class Activation Mapping (Grad-CAM):
-----------------------------------------------------------------------------
1. Hooks forward activations A^k and backward gradients d(y_c)/d(A^k) from the
   deepest convolutional layer of DenseNet-201 / CBAM block.
2. Computes importance weights alpha_k via Global Average Pooling of gradients:
       alpha_k^c = (1 / Z) * sum_i sum_j (d y_c / d A_{i,j}^k)
3. Computes linear combination followed by ReLU:
       L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )
4. Upsamples heatmap to (224x224) and overlays onto the preprocessed lesion image 
   using a JET colormap to illuminate dermoscopic diagnostic structures.
=============================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataset import IDX_TO_LABEL, CLASS_FULL_NAMES


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) tailored for 
    the 3-Brain Hybrid Architecture.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model (nn.Module): The 3-Brain Hybrid Model.
            target_layer (nn.Module, optional): Target convolutional layer.
                                               Defaults to DenseNet201 norm5.
        """
        self.model = model
        self.model.eval()

        # Default to the final batchnorm/conv layer of DenseNet-201
        if target_layer is None:
            self.target_layer = self.model.densenet_features[-1]  # norm5 layer
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for a single image tensor.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed normalized tensor [1, 3, 224, 224].
            target_class (int, optional): Target class index. If None, uses top prediction.
            
        Returns:
            tuple: (heatmap [224, 224] in [0, 1], predicted_class_idx, prediction_confidence)
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        confidence = probs[0, target_class].item()
        
        # Backward pass for the score of the target class
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Gradients: [1, C, H, W] -> Global Average Pooling -> [C]
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation channels
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to isolate features with a POSITIVE contribution
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize between [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        # Resize to original input spatial dimensions (224, 224)
        cam_resized = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)
        return cam_resized, target_class, confidence


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on RGB image.
    
    Args:
        image (np.ndarray): RGB image in [0, 255] uint8.
        cam (np.ndarray): Heatmap in [0, 1] float32.
        alpha (float): Transparency blending weight.
        colormap (int): OpenCV colormap.
        
    Returns:
        np.ndarray: Blended RGB image.
    """
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap_rgb) * alpha + np.float32(image) * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def visualize_gradcam_diagnosis(
    image_preprocessed: np.ndarray,
    cam: np.ndarray,
    predicted_class: int,
    confidence: float,
    ground_truth: Optional[int] = None,
    save_path: Optional[str] = "gradcam_diagnosis.png"
):
    """
    Visualizes raw preprocessed lesion, attention heatmap, and overlay side-by-side.
    """
    overlay = overlay_cam_on_image(image_preprocessed, cam, alpha=0.5)
    pred_label = IDX_TO_LABEL.get(predicted_class, str(predicted_class))
    pred_full = CLASS_FULL_NAMES.get(pred_label, pred_label)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Clean Preprocessed Lesion
    axes[0].imshow(image_preprocessed)
    title_0 = "Preprocessed (DullRazor+CLAHE)"
    if ground_truth is not None:
        gt_label = IDX_TO_LABEL.get(ground_truth, str(ground_truth))
        title_0 += f"\nTrue: {gt_label}"
    axes[0].set_title(title_0, fontsize=11)
    axes[0].axis('off')

    # 2. Grad-CAM Heatmap
    im2 = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("Grad-CAM Attention Map", fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Clinical Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Clinical Overlay\nPred: {pred_label} ({confidence*100:.1f}%)\n{pred_full}", fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[*] Grad-CAM diagnostic visualization saved to: {save_path}")
    plt.close()
