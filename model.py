"""
=============================================================================
Phase 2: The "3-Brain" Hybrid Architecture (DenseNet201 + CBAM + ViT + Fusion)
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Architecture Breakdown:
-----------------------------------------------------------------------------
1. Brain 1 (Local Feature Extractor):
   - Pretrained DenseNet-201 backbone extracting fine-grained multi-scale local 
     features, lesion textures, borders, and color variations.
   - Output Feature Map: [Batch, 1920, 7, 7] (for 224x224 input)

2. Brain 2 (Attention Block - CBAM):
   - Convolutional Block Attention Module placed directly after DenseNet-201.
   - Channel Attention Module (CAM): Recalibrates channel-wise feature responses
     via shared MLP on both AvgPool and MaxPool representations.
   - Spatial Attention Module (SAM): Highlights lesion regions and suppresses 
     healthy skin/vignetting artifacts via 7x7 spatial convolution.
   - Global Pooling -> 1x1920 refined local feature vector.

3. Brain 3 (Global Context Extractor):
   - Pretrained Vision Transformer (ViT-B/16) operating in parallel.
   - 12-layer Multi-Head Self-Attention capturing long-range global contextual 
     dependencies across the entire dermoscopic lesion.
   - Extracts 1x768 Global Feature Vector from the [CLS] token.

4. Feature Fusion & Decision Head:
   - Concatenates Local (1920) + Global (768) -> 1x2688 Fused Representation.
   - Deep Regularized Classification Head with Batch Normalization, Dropout, 
     GELU activations, and 7-class projection.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet201_Weights, ViT_B_16_Weights


# ===========================================================================
# 1. Convolutional Block Attention Module (CBAM)
# ===========================================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) of CBAM.
    
    Exploits inter-channel relationships of features by using both Global Average 
    Pooling and Global Max Pooling followed by a shared Multi-Layer Perceptron (MLP).
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_planes = max(in_planes // reduction_ratio, 16)
        
        # Shared MLP implemented via 1x1 Convolutions
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) of CBAM.
    
    Exploits inter-spatial relationship of features by applying channel-wise pooling
    (Avg + Max) concatenated along channel axis, followed by a 7x7 convolution.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(scale))
        return x * spatial_weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining sequential Channel & Spatial Attention.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ===========================================================================
# 2. "3-Brain" Hybrid Network
# ===========================================================================
class ThreeBrainHybridLesionClassifier(nn.Module):
    """
    State-of-the-Art Hybrid Classifier combining:
    - DenseNet-201 (1920-d local texture extractor)
    - CBAM (Channel & Spatial Attention refinement)
    - ViT-B/16 (768-d global contextual self-attention extractor)
    - Multi-scale Fusion Head (1920 + 768 = 2688 -> 7 classes)
    """
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        freeze_backbones: bool = False
    ):
        super(ThreeBrainHybridLesionClassifier, self).__init__()
        self.num_classes = num_classes

        # -------------------------------------------------------------------
        # Brain 1: DenseNet-201 Local Feature Extractor
        # -------------------------------------------------------------------
        dense_weights = DenseNet201_Weights.DEFAULT if pretrained else None
        base_densenet = models.densenet201(weights=dense_weights)
        self.densenet_features = base_densenet.features
        self.densenet_out_dim = 1920  # DenseNet201 last layer feature dimension

        # -------------------------------------------------------------------
        # Brain 2: CBAM Attention Block
        # -------------------------------------------------------------------
        self.cbAM = CBAM(in_planes=self.densenet_out_dim, reduction_ratio=16, kernel_size=7)
        self.local_pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------------------------------------------------
        # Brain 3: ViT-B/16 Global Context Extractor
        # -------------------------------------------------------------------
        vit_weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = models.vit_b_16(weights=vit_weights)
        self.vit_out_dim = 768  # ViT-B/16 hidden dimension
        
        # Replace classification heads in ViT with Identity to extract raw [CLS] token embedding
        self.vit.heads = nn.Identity()

        if freeze_backbones:
            self._freeze_backbone_weights()

        # -------------------------------------------------------------------
        # Feature Fusion & Classification Head
        # Fused dimension: 1920 (DenseNet+CBAM) + 768 (ViT) = 2688
        # -------------------------------------------------------------------
        self.fused_dim = self.densenet_out_dim + self.vit_out_dim  # 2688
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fused_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

        # Cache for Grad-CAM gradients and activations
        self.gradients = None
        self.activations = None

    def _freeze_backbone_weights(self):
        """Freezes backbone weights for warm-up phase if desired."""
        for param in self.densenet_features.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreezes all parameters for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor):
        """
        Extracts local attention features and global ViT features.
        
        Args:
            x (torch.Tensor): Input batch [B, 3, 224, 224]
            
        Returns:
            tuple: (local_vector [B, 1920], global_vector [B, 768], attention_map [B, 1920, 7, 7])
        """
        # 1. DenseNet201 local features
        dense_feats = self.densenet_features(x)  # [B, 1920, 7, 7]
        
        # 2. CBAM Attention refinement
        refined_feats = self.cbAM(dense_feats)   # [B, 1920, 7, 7]
        self.activations = refined_feats         # Save for Grad-CAM
        
        local_vec = self.local_pool(refined_feats)  # [B, 1920, 1, 1]
        local_vec = torch.flatten(local_vec, 1)     # [B, 1920]

        # 3. ViT Global context features ([CLS] token representation)
        global_vec = self.vit(x)                    # [B, 768]

        return local_vec, global_vec, refined_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass.
        
        Args:
            x (torch.Tensor): Input batch [B, 3, 224, 224]
            
        Returns:
            torch.Tensor: Unnormalized logits [B, 7]
        """
        local_vec, global_vec, _ = self.forward_features(x)
        
        # 4. Feature Fusion: [B, 1920 + 768] = [B, 2688]
        fused = torch.cat([local_vec, global_vec], dim=1)
        
        # 5. Classification
        logits = self.classifier(fused)
        return logits


def build_model(num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.4) -> ThreeBrainHybridLesionClassifier:
    """Convenience factory function."""
    return ThreeBrainHybridLesionClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
