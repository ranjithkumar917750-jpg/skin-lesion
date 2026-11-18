"""
=============================================================================
Main Execution Engine: SOTA Skin Lesion Classification Prototype
=============================================================================
Author: Lead AI Computer Vision Engineer
Project: SOTA Skin Lesion Classification (HAM10000)

Usage:
    python main.py --mode train --data_dir ./data/ham10000 --epochs 40 --batch_size 32
    python main.py --mode eval --checkpoint best_model.pth --data_dir ./data/ham10000
    python main.py --mode demo   # Runs end-to-end pipeline verification on synthetic data
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2

from preprocessing import preprocess_lesion_image, apply_dull_razor, apply_clahe
from model import build_model
from dataset import (
    download_and_extract_ham10000,
    prepare_ham10000_dataframe,
    create_stratified_splits,
    build_dataloaders,
    LABEL_TO_IDX,
    IDX_TO_LABEL
)
from train import fit_model
from evaluate import evaluate_test_set, plot_confusion_matrix, predict_with_mc_dropout
from gradcam import GradCAM, visualize_gradcam_diagnosis


def generate_synthetic_demo_data(demo_dir: str = "./demo_sandbox") -> str:
    """
    Creates synthetic lesion samples for immediate pipeline verification when 
    the full HAM10000 dataset is still downloading.
    """
    os.makedirs(demo_dir, exist_ok=True)
    images_dir = os.path.join(demo_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata = []
    classes = list(LABEL_TO_IDX.keys())
    
    print("[Demo] Generating synthetic dermatoscopy test images with simulated hairs and pigment...")
    for i in range(70):
        dx = classes[i % len(classes)]
        image_id = f"ISIC_DEMO_{i:04d}"
        lesion_id = f"HAM_DEMO_{i // 2:04d}"
        img_path = os.path.join(images_dir, f"{image_id}.jpg")
        
        # Create synthetic skin image with lesion ellipse and simulated hair lines
        img = np.full((300, 300, 3), (210, 180, 170), dtype=np.uint8)  # Skin tone
        # Simulated lesion
        cv2.ellipse(img, (150, 150), (60, 45), 30, 0, 360, (70, 40, 30), -1)
        cv2.circle(img, (140, 145), 25, (40, 20, 15), -1)
        # Simulated dark hairs
        cv2.line(img, (50, 60), (250, 240), (20, 15, 10), 2)
        cv2.line(img, (220, 40), (70, 260), (15, 10, 5), 2)
        
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        metadata.append({
            "lesion_id": lesion_id,
            "image_id": image_id,
            "dx": dx,
            "dx_type": "histo",
            "age": 55,
            "sex": "male",
            "localization": "back"
        })
        
    meta_df = pd.DataFrame(metadata)
    meta_path = os.path.join(demo_dir, "HAM10000_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"[Demo] Synthetic dataset prepared at: {demo_dir}")
    return demo_dir


def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(" SOTA SKIN LESION CLASSIFICATION - '3-BRAIN' HYBRID ARCHITECTURE")
    print(f" Target Compute Device: {device} | PyTorch: {torch.__version__}")
    print("="*70)

    # 1. Dataset Resolution
    if args.mode == "demo":
        dataset_path = generate_synthetic_demo_data("./demo_sandbox")
    else:
        dataset_path = download_and_extract_ham10000(args.data_dir)

    df = prepare_ham10000_dataframe(dataset_path)
    train_df, val_df, test_df = create_stratified_splits(df, val_ratio=0.15, test_ratio=0.15)
    
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224,
        use_weighted_sampler=True
    )

    # 2. Build Model
    model = build_model(num_classes=7, pretrained=True, dropout_rate=0.4).to(device)

    # 3. Train or Load Checkpoint
    if args.mode in ("train", "demo"):
        fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            epochs=args.epochs if args.mode != "demo" else 2,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            patience=args.patience,
            checkpoint_path=args.checkpoint
        )
    elif args.mode == "eval":
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"[Model] Checkpoint successfully loaded: {args.checkpoint}")
        else:
            print(f"[Model] Checkpoint not found at {args.checkpoint}. Evaluating baseline initialized weights.")

    # 4. Evaluation with TTA and Monte Carlo Uncertainty
    results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        use_tta=True,
        use_mc_dropout=True,
        mc_passes=10
    )
    plot_confusion_matrix(results["confusion_matrix"], save_path="confusion_matrix.png")

    # 5. Explainability with Grad-CAM
    print("\n[Grad-CAM] Generating Explainable AI Activation Maps for sample test lesions...")
    sample_img_tensor, sample_target, sample_id = next(iter(test_loader))
    sample_img_tensor = sample_img_tensor[0:1].to(device)
    
    gradcam_engine = GradCAM(model)
    heatmap, pred_cls, conf = gradcam_engine.generate_heatmap(sample_img_tensor)

    # Invert normalization for visualization
    img_unnorm = sample_img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    img_unnorm = (img_unnorm * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_unnorm = np.clip(img_unnorm * 255.0, 0, 255).astype(np.uint8)

    visualize_gradcam_diagnosis(
        image_preprocessed=img_unnorm,
        cam=heatmap,
        predicted_class=pred_cls,
        confidence=conf,
        ground_truth=sample_target[0].item(),
        save_path="gradcam_sample_diagnosis.png"
    )

    print("\n[Completed] SOTA Skin Lesion Pipeline execution finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Skin Lesion Classification - 3-Brain Hybrid Model")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "eval", "demo"],
                        help="Execution mode: train full model, eval checkpoint, or demo verification.")
    parser.add_argument("--data_dir", type=str, default="./data/ham10000", help="Directory containing HAM10000 data")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for pretrained backbones")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Learning rate for CBAM & Classifier head")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Model weights checkpoint file")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loader worker processes")
    args = parser.parse_args()

    run_pipeline(args)
