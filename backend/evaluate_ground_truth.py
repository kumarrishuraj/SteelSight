import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model.model import Unet

# 1. RLE decode function matching notebook make_mask
def make_mask_from_rles(rle_dict, shape=(256, 1600)):
    """
    rle_dict: {1: rle_str_or_nan, 2: ..., 3: ..., 4: ...}
    returns: (4, 256, 1600) float32 numpy array
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for class_id in range(1, 5):
        rle = rle_dict.get(class_id)
        if pd.notna(rle) and isinstance(rle, str):
            label = rle.split()
            positions = map(int, label[0::2])
            lengths = map(int, label[1::2])
            mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            for pos, le in zip(positions, lengths):
                # RLE is 1-based index
                mask[pos - 1 : pos - 1 + le] = 1
            masks[:, :, class_id - 1] = mask.reshape(shape[0], shape[1], order='F')
    return masks.transpose(2, 0, 1) # (4, 256, 1600)

def dice_score(pred, truth, eps=1e-7):
    # Both shape: (H, W) or flattened binary {0, 1}
    p_sum = pred.sum()
    t_sum = truth.sum()
    if p_sum == 0 and t_sum == 0:
        return 1.0 # True negative
    if p_sum == 0 or t_sum == 0:
        return 0.0
    intersection = (pred * truth).sum()
    return float((2.0 * intersection) / (p_sum + t_sum + eps))

def iou_score(pred, truth, eps=1e-7):
    p_sum = pred.sum()
    t_sum = truth.sum()
    if p_sum == 0 and t_sum == 0:
        return 1.0 # True negative
    intersection = (pred * truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))

def main():
    print("Loading model checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("model/model.pth", map_location=device, weights_only=False)
    
    net = Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        decoder_channels=(256, 128, 64, 32, 16),
        classes=4,
        activation=None
    )
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.to(device)
    net.eval()
    print(f"Model loaded successfully on {device}. Checkpoint epoch: {checkpoint.get('epoch')}")

    # Load dataset
    df = pd.read_csv("dataset/train.csv")
    pivoted = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    
    # Stratified validation split reproducible seed=69 (matching notebook)
    # The notebook did:
    # df['defects'] = df.count(axis=1)
    # train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    from sklearn.model_selection import train_test_split
    pivoted['defects'] = pivoted.count(axis=1)
    train_df, val_df = train_test_split(pivoted, test_size=0.2, stratify=pivoted["defects"], random_state=69)
    print(f"Validation set size: {len(val_df)} images with annotations")

    # Evaluate on a representative sample of validation images (e.g., 50 images with defects)
    val_sample = val_df.head(60)
    
    class_dices = {1: [], 2: [], 3: [], 4: []}
    class_ious = {1: [], 2: [], 3: [], 4: []}
    overall_dices = []
    overall_ious = []
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    count = 0
    for image_id, row in val_sample.iterrows():
        img_path = os.path.join("dataset/train_images", image_id)
        if not os.path.exists(img_path):
            continue
        
        # Read and preprocess
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Ground truth mask (4, 256, 1600)
        gt_masks = make_mask_from_rles(row.to_dict())
        
        # Predict
        norm_img = ((img_rgb.astype(np.float32) / 255.0) - mean) / std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = net(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            pred_masks = (probs > 0.5).astype(np.float32)
        
        # Compute per class Dice & IoU
        for c in range(4):
            d = dice_score(pred_masks[c], gt_masks[c])
            iu = iou_score(pred_masks[c], gt_masks[c])
            class_dices[c+1].append(d)
            class_ious[c+1].append(iu)
        
        # Overall image mean
        overall_dices.append(np.mean([class_dices[c][count] for c in range(1, 5)]))
        overall_ious.append(np.mean([class_ious[c][count] for c in range(1, 5)]))
        count += 1

    print(f"\n--- Validation Results on {count} Test Samples (Threshold = 0.5) ---")
    for c in range(1, 5):
        mean_dice = np.mean(class_dices[c])
        mean_iou = np.mean(class_ious[c])
        print(f"Class {c} ({['Defect Class 1', 'Defect Class 2', 'Defect Class 3', 'Defect Class 4'][c-1]}): Dice = {mean_dice:.4f} | IoU = {mean_iou:.4f}")
    
    print(f"\nOverall Mean Dice: {np.mean(overall_dices):.4f}")
    print(f"Overall Mean IoU:  {np.mean(overall_ious):.4f}")

if __name__ == "__main__":
    main()
