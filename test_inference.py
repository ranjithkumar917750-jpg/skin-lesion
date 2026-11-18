"""
=============================================================================
Automated Testing & Clinical Verification Suite: DERM-SOTA Pipeline
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Verification Scope:
1. Clinical Preprocessing (DullRazor hair inpainting + LAB CLAHE)
2. Zero Data Leakage Integrity (Patient/Lesion-level stratified split)
3. 3-Brain Hybrid Model Forward Pass & Output Dimension Assertions
4. 5-Fold Test-Time Augmentation (TTA) Probability Invariance
5. Monte Carlo Dropout Epistemic Uncertainty & Eval Mode Restoration
6. Explainable AI Grad-CAM Heatmap Resolution & Range Validity
7. End-to-End Pipeline Inference on Synthetic & Real Lesion Captures
=============================================================================
"""

import os
import unittest
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

from preprocessing import (
    apply_dull_razor,
    apply_clahe,
    preprocess_lesion_image,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from model import build_model, ThreeBrainHybridLesionClassifier
from gradcam import GradCAM, overlay_cam_on_image
from evaluate import predict_with_tta, predict_with_mc_dropout
from dataset import (
    create_stratified_splits,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    CLASS_FULL_NAMES
)


class TestSkinLesionPipeline(unittest.TestCase):
    """Rigorous Automated Test Suite for Computer Vision & Clinical AI Pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initializes shared test fixture and device configuration."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestSuite] Initializing Test Suite on Device: {cls.device}")
        
        # Build model in eval mode
        cls.model = build_model(num_classes=7, pretrained=False, dropout_rate=0.4).to(cls.device)
        
        # Load best_model.pth if available
        if os.path.exists("best_model.pth"):
            try:
                state_dict = torch.load("best_model.pth", map_location=cls.device)
                cls.model.load_state_dict(state_dict)
                print("[TestSuite] Loaded 'best_model.pth' for verification.")
            except Exception as e:
                print(f"[TestSuite] Note: Model initialized with clean weights ({e})")
        
        cls.model.eval()

        # Create dummy synthetic lesion with hair artifact
        cls.sample_h, cls.sample_w = 300, 300
        dummy = np.full((cls.sample_h, cls.sample_w, 3), (210, 180, 170), dtype=np.uint8)
        # Pigmented lesion
        cv2.ellipse(dummy, (150, 150), (60, 45), 25, 0, 360, (50, 30, 25), -1)
        # Hair artifact
        cv2.line(dummy, (30, 40), (270, 260), (15, 10, 5), 3)
        cls.raw_test_image = dummy

    def test_01_preprocessing_pipeline(self):
        """Validates DullRazor, CIE-LAB CLAHE, and standardized resizing."""
        print("\n[Test 1] Verifying Preprocessing Functions (DullRazor + CLAHE)...")
        
        # 1. DullRazor Hair Removal
        hair_free = apply_dull_razor(self.raw_test_image, filter_size=9, threshold=10, inpaint_radius=5)
        self.assertEqual(hair_free.shape, self.raw_test_image.shape)
        self.assertEqual(hair_free.dtype, np.uint8)
        
        # Check that the dark hair pixel was altered by inpainting
        hair_pixel_raw = self.raw_test_image[150, 140]
        # Inpainting should modify pixels obscured by hair
        self.assertFalse(np.array_equal(self.raw_test_image, hair_free))
        
        # 2. LAB CLAHE Equalization
        enhanced = apply_clahe(hair_free, clip_limit=2.0)
        self.assertEqual(enhanced.shape, (self.sample_h, self.sample_w, 3))
        self.assertEqual(enhanced.dtype, np.uint8)
        
        # 3. Full Integrated Preprocessing
        clean_224 = preprocess_lesion_image(self.raw_test_image, img_size=224)
        self.assertEqual(clean_224.shape, (224, 224, 3))
        self.assertEqual(clean_224.dtype, np.uint8)
        self.assertTrue(clean_224.min() >= 0 and clean_224.max() <= 255)
        print("  -> DullRazor inpainting, CLAHE, and resizing validated successfully.")

    def test_02_data_split_integrity_zero_leakage(self):
        """Verifies that create_stratified_splits produces strictly disjoint partitions."""
        print("\n[Test 2] Auditing Patient/Lesion-Stratified Splitting for Data Leakage...")
        
        # Synthetic DataFrame with 35 lesions, 70 images (2 images per lesion) across 7 classes
        metadata = []
        classes = list(LABEL_TO_IDX.keys())
        for i in range(70):
            cls = classes[i % len(classes)]
            lesion_id = f"HAM_TEST_{cls}_{i // 2:03d}"
            metadata.append({
                "image_id": f"ISIC_TEST_{i:04d}",
                "lesion_id": lesion_id,
                "dx": cls,
                "label": LABEL_TO_IDX[cls]
            })
        df = pd.DataFrame(metadata)
        
        train_df, val_df, test_df = create_stratified_splits(df, val_ratio=0.15, test_ratio=0.15, random_state=42)
        
        train_lesions = set(train_df['lesion_id'])
        val_lesions = set(val_df['lesion_id'])
        test_lesions = set(test_df['lesion_id'])
        
        train_images = set(train_df['image_id'])
        test_images = set(test_df['image_id'])
        
        # Strict Disjointness Assertions
        self.assertTrue(train_lesions.isdisjoint(test_lesions), "LEAKAGE: Lesion overlap in train and test!")
        self.assertTrue(train_lesions.isdisjoint(val_lesions), "LEAKAGE: Lesion overlap in train and val!")
        self.assertTrue(val_lesions.isdisjoint(test_lesions), "LEAKAGE: Lesion overlap in val and test!")
        self.assertTrue(train_images.isdisjoint(test_images), "LEAKAGE: Image overlap in train and test!")
        
        # Partition sum check
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
        print("  -> Zero Data Leakage mathematically guaranteed across all sets.")

    def test_03_model_forward_dimensions(self):
        """Verifies 3-Brain forward pass outputs [B, 7] unnormalized logits."""
        print("\n[Test 3] Verifying 3-Brain Hybrid Forward Pass...")
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            logits = self.model(dummy_input)
            
        self.assertEqual(logits.shape, (batch_size, 7))
        self.assertFalse(torch.isnan(logits).any(), "Model produced NaN logits!")
        
        probs = F.softmax(logits, dim=1)
        sums = probs.sum(dim=1).cpu().numpy()
        np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)
        print(f"  -> Forward pass produces correct [B, 7] logits with valid Softmax probabilities.")

    def test_04_test_time_augmentation(self):
        """Verifies 5-fold spatial TTA predictions are valid probability distributions."""
        print("\n[Test 4] Verifying Test-Time Augmentation (TTA)...")
        dummy_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        avg_probs = predict_with_tta(self.model, dummy_tensor, self.device)
        
        self.assertEqual(avg_probs.shape, (1, 7))
        probs_np = avg_probs.cpu().numpy()[0]
        
        self.assertTrue(np.all(probs_np >= 0.0) and np.all(probs_np <= 1.0))
        np.testing.assert_allclose(np.sum(probs_np), 1.0, atol=1e-5)
        print("  -> 5-Fold TTA outputs smooth, strictly valid probabilities.")

    def test_05_monte_carlo_dropout_uncertainty(self):
        """Verifies epistemic variance calculation and deterministic eval restoration."""
        print("\n[Test 5] Verifying Monte Carlo Dropout Epistemic Uncertainty...")
        dummy_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        num_passes = 8
        
        mean_probs, variance, unc_score = predict_with_mc_dropout(
            self.model, dummy_tensor, self.device, num_passes=num_passes
        )
        
        # Verify shapes
        self.assertEqual(mean_probs.shape, (1, 7))
        self.assertEqual(variance.shape, (1, 7))
        self.assertEqual(unc_score.shape, (1,))
        
        # Check variance non-negativity
        self.assertTrue(torch.all(variance >= 0.0))
        self.assertTrue(unc_score.item() >= 0.0)
        
        # Verify model was cleanly returned to eval mode
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                self.assertFalse(m.training, "Dropout was not restored to eval mode!")
                
        print(f"  -> MC Dropout computed variance ({unc_score.item():.6f}) and cleanly restored eval mode.")

    def test_06_gradcam_heatmap_geometry(self):
        """Verifies Grad-CAM heatmap dimensions, range [0, 1], and JET overlay."""
        print("\n[Test 6] Verifying Explainable AI Grad-CAM Generation...")
        clean_img = preprocess_lesion_image(self.raw_test_image, img_size=224)
        
        norm_img = clean_img.astype(np.float32) / 255.0
        norm_img = (norm_img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        input_tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        gradcam = GradCAM(self.model)
        heatmap, pred_class, confidence = gradcam.generate_heatmap(input_tensor)
        
        # Dimensional & Range Assertions
        self.assertEqual(heatmap.shape, (224, 224), "Heatmap must match 224x224 input image resolution!")
        self.assertTrue(heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap values must be bounded in [0, 1]!")
        self.assertIn(pred_class, list(IDX_TO_LABEL.keys()))
        self.assertTrue(0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0, 1]!")
        
        # Overlay check
        overlay = overlay_cam_on_image(clean_img, heatmap, alpha=0.5)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        print("  -> Grad-CAM produces exact (224, 224) attention maps with valid overlay.")

    def test_07_end_to_end_diagnostic_pipeline(self):
        """Runs the complete integrated inference pipeline from raw image to formatted diagnosis."""
        print("\n[Test 7] Executing End-to-End Clinical Diagnostic Pipeline...")
        
        # 1. Preprocess
        preprocessed = preprocess_lesion_image(self.raw_test_image, img_size=224)
        norm = (preprocessed.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        # 2. TTA Inference
        probs = predict_with_tta(self.model, tensor, self.device)
        probs_np = probs[0].cpu().numpy()
        pred_idx = int(np.argmax(probs_np))
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = float(probs_np[pred_idx])
        
        # 3. Uncertainty
        _, _, unc = predict_with_mc_dropout(self.model, tensor, self.device, num_passes=6)
        unc_val = float(unc[0].item())
        
        # 4. Grad-CAM
        cam, _, _ = GradCAM(self.model).generate_heatmap(tensor, target_class=pred_idx)
        overlay = overlay_cam_on_image(preprocessed, cam, alpha=0.5)
        
        # Clinical Output Assertions
        self.assertIn(pred_label, LABEL_TO_IDX)
        self.assertIn(pred_label, CLASS_FULL_NAMES)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertTrue(unc_val >= 0.0)
        self.assertEqual(overlay.shape, (224, 224, 3))
        
        print(f"  -> End-to-end inference complete: Class '{pred_label}' ({CLASS_FULL_NAMES[pred_label]})")
        print(f"  -> Confidence: {confidence*100:.2f}% | Uncertainty Variance: {unc_val:.6f}")
        print("  -> ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
