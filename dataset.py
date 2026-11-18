"""
=============================================================================
Phase 1 & 3: HAM10000 Dataset Loader, Automated Ingestion & Split Management
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Features:
- Automated dataset acquisition via kagglehub / kaggle API / fallback.
- Lesion-level Stratified Splitting to eliminate data leakage between train/val/test.
- DullRazor + CLAHE integrated directly into PyTorch Dataset pipeline.
- WeightedRandomSampler support to actively equalize mini-batch class frequencies.
=============================================================================
"""

import os
import glob
import shutil
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from preprocessing import preprocess_lesion_image, get_transforms


# Canonical HAM10000 7 Diagnostic Classes Mapping
LABEL_TO_IDX: Dict[str, int] = {
    'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
    'bcc':   1,  # Basal cell carcinoma
    'bkl':   2,  # Benign keratosis-like lesions
    'df':    3,  # Dermatofibroma
    'mel':   4,  # Melanoma
    'nv':    5,  # Melanocytic nevi
    'vasc':  6   # Vascular lesions
}

IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}

CLASS_FULL_NAMES: Dict[str, str] = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis-like Lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}


def download_and_extract_ham10000(dest_dir: str = "./data/ham10000") -> str:
    """
    Automated acquisition and extraction pipeline for the HAM10000 dataset.
    
    Checks local existence first; if missing, triggers download via kagglehub
    or official Kaggle CLI.
    
    Args:
        dest_dir (str): Root destination directory.
        
    Returns:
        str: Absolute path to the resolved dataset root.
    """
    os.makedirs(dest_dir, exist_ok=True)
    metadata_candidates = glob.glob(os.path.join(dest_dir, "**", "*metadata*.csv"), recursive=True)
    
    if metadata_candidates:
        print(f"[Dataset] HAM10000 metadata discovered locally: {metadata_candidates[0]}")
        return dest_dir

    print("[Dataset] HAM10000 not found locally. Initiating automated acquisition...")
    
    # Strategy 1: kagglehub (Google/Kaggle official lightweight downloader)
    try:
        import kagglehub
        print("[Dataset] Attempting download via kagglehub...")
        download_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"[Dataset] Successfully downloaded to cache: {download_path}")
        
        # Link or copy into dest_dir
        for item in os.listdir(download_path):
            s = os.path.join(download_path, item)
            d = os.path.join(dest_dir, item)
            if not os.path.exists(d):
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        print(f"[Dataset] Assets staged in: {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"[Dataset] kagglehub acquisition skipped/failed: {e}")

    # Strategy 2: Kaggle CLI API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        print("[Dataset] Attempting download via Kaggle CLI API...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("kmader/skin-cancer-mnist-ham10000", path=dest_dir, unzip=True)
        print(f"[Dataset] Kaggle CLI download and extraction complete: {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"[Dataset] Kaggle API failed: {e}")
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTION:")
        print("1. Download 'skin-cancer-mnist-ham10000' from Kaggle or Harvard Dataverse:")
        print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"2. Unzip all contents into: {os.path.abspath(dest_dir)}")
        print("="*70 + "\n")
        return dest_dir


def prepare_ham10000_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Indexes all image files, merges with HAM10000 metadata, and creates a clean DataFrame.
    
    Args:
        data_dir (str): Root dataset folder.
        
    Returns:
        pd.DataFrame: Clean DataFrame with columns [image_id, lesion_id, dx, path, label].
    """
    metadata_files = glob.glob(os.path.join(data_dir, "**", "*metadata*.csv"), recursive=True)
    if not metadata_files:
        raise FileNotFoundError(f"Could not locate HAM10000_metadata.csv inside {data_dir}")
    
    meta_df = pd.read_csv(metadata_files[0])
    
    # Map all image files in all subdirectories (part 1, part 2, etc.)
    all_image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    if not all_image_paths:
        raise FileNotFoundError(f"No .jpg images found inside {data_dir}")
    
    image_path_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in all_image_paths
    }
    
    meta_df['path'] = meta_df['image_id'].map(image_path_map)
    # Drop records where image file is not present
    meta_df = meta_df.dropna(subset=['path']).reset_index(drop=True)
    
    # Encode target label integers
    meta_df['label'] = meta_df['dx'].map(LABEL_TO_IDX)
    meta_df = meta_df.dropna(subset=['label']).reset_index(drop=True)
    meta_df['label'] = meta_df['label'].astype(int)
    
    print(f"[Dataset] Indexed {len(meta_df)} dermoscopy images across 7 diagnostic classes.")
    return meta_df


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 with DullRazor + CLAHE preprocessing & Albumentations.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        apply_clinical_preprocess: bool = True,
        img_size: int = 224
    ):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing 'path' and 'label'.
            transform: Albumentations Compose pipeline.
            apply_clinical_preprocess (bool): Whether to run DullRazor + CLAHE.
            img_size (int): Image size to format into.
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.apply_clinical_preprocess = apply_clinical_preprocess
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['label']

        # Load RGB image via PIL -> np.ndarray
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)

        # Apply Phase 1 Preprocessing: DullRazor hair removal + CLAHE
        if self.apply_clinical_preprocess:
            img_np = preprocess_lesion_image(img_np, img_size=self.img_size)

        # Apply Phase 1/3 Albumentations augmentations / normalization
        if self.transform is not None:
            augmented = self.transform(image=img_np)
            tensor_img = augmented['image']
        else:
            # Fallback simple tensor conversion
            tensor_img = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0

        return tensor_img, torch.tensor(label, dtype=torch.long), row['image_id']


def create_stratified_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs Patient/Lesion-aware Stratified Splitting.
    
    Crucial for dermatological integrity: If the same lesion was photographed multiple 
    times, putting one in Train and one in Test artificially inflates test accuracy.
    This function groups by lesion_id while preserving class proportions.
    """
    # Group by lesion_id and obtain dominant class per lesion
    lesion_df = df.groupby('lesion_id').agg({'dx': 'first', 'label': 'first'}).reset_index()
    
    # Check if stratified splitting is feasible (test set size must be >= num_classes and min class count >= 2)
    n_classes = len(lesion_df['label'].unique())
    min_class_count = lesion_df['label'].value_counts().min()
    can_stratify_test = (int(len(lesion_df) * test_ratio) >= n_classes) and (min_class_count >= 2)
    stratify_target = lesion_df['label'] if can_stratify_test else None

    # First split: Train+Val vs Test
    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=test_ratio,
        stratify=stratify_target,
        random_state=random_state
    )
    
    # Second split: Train vs Val
    val_adjusted_ratio = val_ratio / (1.0 - test_ratio)
    n_tv_classes = len(train_val_lesions['label'].unique())
    can_stratify_val = (int(len(train_val_lesions) * val_adjusted_ratio) >= n_tv_classes) and (train_val_lesions['label'].value_counts().min() >= 2)
    stratify_train_val = train_val_lesions['label'] if can_stratify_val else None
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=val_adjusted_ratio,
        stratify=stratify_train_val,
        random_state=random_state
    )
    
    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
    val_df   = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
    test_df  = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)
    
    # Strict Mathematical Verification: Zero Lesion or Image Leakage
    train_lesions_set = set(train_df['lesion_id'])
    val_lesions_set = set(val_df['lesion_id'])
    test_lesions_set = set(test_df['lesion_id'])
    
    train_images_set = set(train_df['image_id'])
    val_images_set = set(val_df['image_id'])
    test_images_set = set(test_df['image_id'])
    
    assert train_lesions_set.isdisjoint(test_lesions_set), (
        f"[DATA LEAKAGE ERROR] Train and Test share {len(train_lesions_set.intersection(test_lesions_set))} lesions!"
    )
    assert train_lesions_set.isdisjoint(val_lesions_set), (
        f"[DATA LEAKAGE ERROR] Train and Val share {len(train_lesions_set.intersection(val_lesions_set))} lesions!"
    )
    assert val_lesions_set.isdisjoint(test_lesions_set), (
        f"[DATA LEAKAGE ERROR] Val and Test share {len(val_lesions_set.intersection(test_lesions_set))} lesions!"
    )
    assert train_images_set.isdisjoint(test_images_set), "[DATA LEAKAGE ERROR] Image overlap between Train and Test!"
    
    print("=" * 75)
    print("           PATIENT/LESION-STRATIFIED SPLIT INTEGRITY VERIFIED")
    print("=" * 75)
    print(f"Total Lesions:  {len(lesion_df)} (Train: {len(train_lesions_set)}, Val: {len(val_lesions_set)}, Test: {len(test_lesions_set)})")
    print(f"Total Images:   {len(df)} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    print(f"Disjoint Check: Zero lesion overlap between Train, Val, and Test sets (Strictly 0%).")
    print("=" * 75)
    return train_df, val_df, test_df


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Constructs PyTorch DataLoaders with Albumentations pipelines and optional 
    WeightedRandomSampler for minority oversampling.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    train_transform, val_transform = get_transforms(img_size=img_size)

    train_dataset = HAM10000Dataset(train_df, transform=train_transform, apply_clinical_preprocess=True, img_size=img_size)
    val_dataset   = HAM10000Dataset(val_df, transform=val_transform, apply_clinical_preprocess=True, img_size=img_size)
    test_dataset  = HAM10000Dataset(test_df, transform=val_transform, apply_clinical_preprocess=True, img_size=img_size)

    # Calculate class distribution
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Setup mini-batch sampler to equalize minority classes during training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = [class_weights[label] for label in train_df['label']]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_weights_tensor
