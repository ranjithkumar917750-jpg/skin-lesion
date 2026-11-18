"""
=============================================================================
Phase 1: Advanced Preprocessing Pipeline & Data Augmentation
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Pipeline:
1. DullRazor Morphological Hair Removal (Black-Hat Filtering + Masking + Inpainting)
2. Lighting Correction via CLAHE on LAB Color Space (Preserves Color Fidelity)
3. Geometric & Color Augmentations via Albumentations
4. Standardization: Resize (224x224) & ImageNet Normalization
=============================================================================
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics for transfer learning backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_dull_razor(image: np.ndarray, filter_size: int = 9, threshold: int = 10, inpaint_radius: int = 5) -> np.ndarray:
    """
    Morphological hair removal algorithm inspired by DullRazor.
    
    Workflow:
    1. Grayscale conversion.
    2. Black-Hat morphological filtering to isolate dark, narrow hair filaments.
    3. Binary thresholding to generate hair mask.
    4. Telea inpainting to reconstruct skin surface behind detected hairs.
    
    Args:
        image (np.ndarray): Input RGB image of shape (H, W, 3) with uint8 values [0, 255].
        filter_size (int): Structuring element kernel size for morphological filter.
        threshold (int): Intensity difference threshold for hair mask generation.
        inpaint_radius (int): Pixel radius for Fast Marching Method inpainting.
        
    Returns:
        np.ndarray: Hair-removed RGB image.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image passed to apply_dull_razor.")
    
    # 1. Convert to grayscale for morphological filtering
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Black-Hat filter highlights structures darker than their surroundings
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_size, filter_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Threshold to obtain hair mask
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # Optional morphological closing on mask to bridge small gaps in detected hairs
    mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, mask_kernel)
    
    # 4. Inpainting to restore skin texture underneath hairs
    hair_removed = cv2.inpaint(image, hair_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return hair_removed


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) in CIE-LAB color space.
    
    Applying CLAHE directly in RGB causes severe unnatural color shifts, which alters 
    diagnostic features like erythema (redness) and melanin pigmentation. By transforming
    to CIE-LAB and equalizing ONLY the L (Luminance) channel, we enhance local edge 
    and texture contrast while strictly preserving clinical chrominance (A & B channels).
    
    Args:
        image (np.ndarray): Input RGB image [0, 255].
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Grid dimension for contextual regions (default: 8x8).
        
    Returns:
        np.ndarray: Contrast-enhanced RGB image.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image passed to apply_clahe.")
    
    # Convert RGB to CIE-LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_lesion_image(image: np.ndarray, img_size: int = 224) -> np.ndarray:
    """
    Full clinical preprocessing pipeline combining DullRazor + CLAHE.
    
    Args:
        image (np.ndarray): Raw dermoscopy RGB image.
        img_size (int): Target square dimension.
        
    Returns:
        np.ndarray: Cleaned, contrast-adjusted, resized RGB image (uint8, 0-255).
    """
    # Step 1: DullRazor hair removal
    hair_free = apply_dull_razor(image, filter_size=9, threshold=10, inpaint_radius=5)
    
    # Step 2: Lighting correction via CLAHE
    enhanced = apply_clahe(hair_free, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 3: Resize
    if enhanced.shape[0] != img_size or enhanced.shape[1] != img_size:
        enhanced = cv2.resize(enhanced, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        
    return enhanced


def get_transforms(img_size: int = 224):
    """
    Builds Albumentations transformation pipelines for training and validation/testing.
    
    Training pipeline features heavy spatial and photometric augmentations to artificially
    inflate minority classes and avoid overfitting on benign nevi.
    
    Args:
        img_size (int): Image spatial resolution (224 for DenseNet/ViT).
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(12, 28),
            hole_width_range=(12, 28),
            fill=0,
            p=0.4
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    return train_transform, val_transform
