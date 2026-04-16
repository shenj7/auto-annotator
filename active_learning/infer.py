"""
infer.py — Run the trained model on a full TEM image and display detections.

Usage:
    python active_learning/infer.py --image path/to/image.png
    python active_learning/infer.py --image path/to/image.png --threshold 0.5
    python active_learning/infer.py --image path/to/image.png --save out.png

The script:
  1. Loads the latest checkpoint from active_learning_data/checkpoints/model.pt
  2. Splits the image into 128x128 patches (same as the training pipeline)
  3. Runs each patch through the model (MC Dropout, 10 passes for uncertainty)
  4. Stitches the probability maps back into a full-image heatmap
  5. Thresholds into a binary mask and extracts contours
  6. Displays the original image with green contours overlaid
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow running from repo root or from active_learning/
sys.path.insert(0, str(Path(__file__).parent))

from model import ResAttentionUNet
from patch_utils import extract_patches, merge_patches

PATCH_SIZE      = 128
CHECKPOINT_PATH = Path("active_learning_data/checkpoints/model.pt")


def load_model(device: str) -> ResAttentionUNet:
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}.")
        print("Train the model first by running:  python active_learning/main.py --train")
        sys.exit(1)

    model = ResAttentionUNet(n_channels=1, n_classes=1, dropout_rate=0.3)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"Loaded checkpoint — round {checkpoint.get('round', '?')}, "
          f"trained on {checkpoint.get('num_verified', '?')} verified samples.")
    return model


def run_inference(model: ResAttentionUNet, img_gray: np.ndarray, device: str,
                  n_passes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        prob_map:  (H, W) float32 in [0, 1] — averaged sigmoid probability
        unc_map:   (H, W) float32 — pixel-wise entropy (uncertainty)
    """
    h, w = img_gray.shape
    patches = extract_patches(img_gray,
                              patch_size=(PATCH_SIZE, PATCH_SIZE),
                              stride=(PATCH_SIZE, PATCH_SIZE))

    prob_patches = []
    unc_patches  = []

    model.eval()
    model.enable_dropout()

    for patch_img, x, y in patches:
        tensor = torch.from_numpy(patch_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(tensor)
                preds.append(torch.sigmoid(logits))

        preds      = torch.stack(preds)                          # (N, 1, 1, H, W)
        mean_probs = torch.mean(preds, dim=0).squeeze().cpu().numpy()

        eps     = 1e-7
        entropy = (-mean_probs * np.log(mean_probs + eps)
                   - (1 - mean_probs) * np.log(1 - mean_probs + eps))

        prob_patches.append((mean_probs, x, y))
        unc_patches.append((entropy,     x, y))

    prob_map = merge_patches(prob_patches, (h, w)).astype(np.float32)
    unc_map  = merge_patches(unc_patches,  (h, w)).astype(np.float32)
    return prob_map, unc_map


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image",     required=True, help="Path to the TEM image.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for calling a pixel a dislocation.")
    parser.add_argument("--passes",    type=int,   default=10,
                        help="Number of MC Dropout forward passes.")
    parser.add_argument("--save",      default=None,
                        help="If given, save the overlay image to this path instead of displaying.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    print(f"Image: {img_path.name}  {img_gray.shape[1]}×{img_gray.shape[0]} px")
    print(f"Device: {device} | Threshold: {args.threshold} | MC passes: {args.passes}")

    model    = load_model(device)
    prob_map, unc_map = run_inference(model, img_gray, device, n_passes=args.passes)

    # Binary mask + contours
    binary = (prob_map >= args.threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detections: {len(contours)} contours at threshold {args.threshold}")

    # Build overlay
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(img_path.name, fontsize=13)

    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Detections (thresh={args.threshold})  —  {len(contours)} found")
    axes[1].axis("off")

    im = axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Probability map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
