import math
import os
import shutil
import argparse
import cv2
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np

from bootstrap import ClassicalPipeline
from model import ResAttentionUNet
from active_learning import SelectionEngine
from oracle_ui import OracleUI
from trainer import VerifiedDataset, FeedbackRefinement, train_model

# Defaults — override any of these from the command line (see --help)
DEFAULT_MIN_SIZE    = 0.00025  # fraction of image area (~0.003 %)
DEFAULT_MAX_SIZE    = 0.00144  # fraction of image area (~0.040 %)
DEFAULT_CONTRAST    = 30        # min local contrast (0–255); raise to reduce false positives
DEFAULT_SCALE       = 5.0      # background Gaussian sigma (px); should exceed blob radius
DEFAULT_PRE_SMOOTH  = 1.0      # pre-smoothing sigma (px)
DEFAULT_CIRCULARITY = 0.3      # min circularity (0 = off, 1 = perfect circle)
DEFAULT_DILATION    = 0        # pixels to expand each contour outward (0 = off)


def make_algorithm(min_size, max_size, contrast, scale, pre_smooth, circularity, dilation):
    """
    Returns a bootstrap detection function with all parameters baked in.

    Uses local-contrast thresholding (much more robust than global Otsu for TEM):
      1. Pre-smooth to suppress sensor noise.
      2. Blur at `scale` sigma to estimate the local background.
      3. Subtract → pixels darker than their neighbourhood become positive.
      4. Threshold by `contrast` → binary foreground map.
      5. Morphological opening to remove isolated speckle.
      6. Filter contours by area and circularity.
      7. Dilate accepted contours outward by `dilation` pixels to capture
         the full extent of the gradient halo around each feature.
    """
    def algorithm(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        total_px = h * w
        min_area = int(min_size * total_px)
        max_area = int(max_size * total_px)

        smoothed   = cv2.GaussianBlur(img,      (0, 0), sigmaX=pre_smooth, sigmaY=pre_smooth)
        background = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=scale,      sigmaY=scale)

        # Positive where a pixel is darker than its local neighbourhood
        diff = cv2.subtract(background, smoothed)
        _, binary = cv2.threshold(diff, int(contrast), 255, cv2.THRESH_BINARY)

        # Remove isolated single-pixel speckle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue
            if circularity > 0:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                if (4 * math.pi * area / perimeter ** 2) < circularity:
                    continue
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

        # Expand each accepted region outward to capture the gradient halo
        if dilation > 0:
            dil_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
            mask = cv2.dilate(mask, dil_kernel, iterations=1)

        return mask

    return algorithm

def main(args):
    # 1. Define Directories
    BASE_DIR = Path("active_learning_data")
    RAW_DIR = BASE_DIR / "raw_data"
    UNLABELED_DIR = BASE_DIR / "unlabeled"
    PENDING_DIR = BASE_DIR / "pending_review"
    VERIFIED_DIR = BASE_DIR / "verified_dataset"
    REJECTED_DIR = BASE_DIR / "rejected"

    for d in [RAW_DIR, UNLABELED_DIR, PENDING_DIR, VERIFIED_DIR, REJECTED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check phase
    num_unlabeled = len(list(UNLABELED_DIR.glob("*.*")))
    num_verified = len(list((VERIFIED_DIR / "images").glob("*.*")))
    num_pending = len(list((PENDING_DIR / "images").glob("*.*")))

    print("--- Active Learning Loop Status ---")
    print(f"Verified: {num_verified} | Unlabeled: {num_unlabeled} | Pending Review: {num_pending}")
    
    # 2. Bootstrap if needed
    if num_pending == 0 and num_verified == 0:
        print("Bootstrapping to generate first proposals...")
        print(f"  min_size={args.min_size:.5f}  max_size={args.max_size:.5f}  "
              f"contrast={args.contrast}  scale={args.scale}  "
              f"pre_smooth={args.pre_smooth}  circularity={args.circularity}")
        algorithm = make_algorithm(
            min_size=args.min_size,
            max_size=args.max_size,
            contrast=args.contrast,
            scale=args.scale,
            pre_smooth=args.pre_smooth,
            circularity=args.circularity,
            dilation=args.dilation,
        )
        pipeline = ClassicalPipeline(
            raw_data_dir=str(RAW_DIR),
            output_dir=str(PENDING_DIR),
            algorithm_fn=algorithm
        )
        pipeline.process_all()
        print("Bootstrap complete. Please run the UI to verify initial proposals.")
        return

    # --train: move any remaining pending images back to unlabeled, then train
    if args.train and num_pending > 0:
        pending_images = list((PENDING_DIR / "images").glob("*.*"))
        for img_path in pending_images:
            shutil.move(str(img_path), str(UNLABELED_DIR / img_path.name))
        # Clean up leftover masks/overlays for moved images
        for subdir in ("masks", "overlays"):
            for f in (PENDING_DIR / subdir).glob("*.*"):
                f.unlink(missing_ok=True)
        moved = len(pending_images)
        print(f"Moved {moved} unannotated pending image(s) back to unlabeled.")
        num_pending = 0
        num_unlabeled += moved

    # 3. If there are verified samples, Train Model and run Active Learning
    if num_verified > 0 and num_pending == 0 and num_unlabeled > 0:
        CHECKPOINT_DIR = BASE_DIR / "checkpoints"
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"

        print("\n--- Training Model ---")
        dataset = VerifiedDataset(str(VERIFIED_DIR))
        model = ResAttentionUNet(n_channels=1, n_classes=1, dropout_rate=0.3)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Resume from checkpoint if one exists
        if CHECKPOINT_PATH.exists():
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Resumed from checkpoint (round {checkpoint.get('round', '?')}, "
                  f"{checkpoint.get('num_verified', '?')} verified samples).")
        else:
            print("No checkpoint found — training from scratch.")

        refiner = FeedbackRefinement(model, optimizer, device)

        # Seed replay buffer with all currently verified examples
        added = refiner.add_approved_from_dir(VERIFIED_DIR)
        if added:
            print(f"Loaded {added} verified examples into replay buffer.")

        # Apply contrastive steps for any previously rejected proposals
        penalised = refiner.apply_contrastive_from_dir(REJECTED_DIR)
        if penalised:
            print(f"Applied contrastive penalty for {penalised} rejected proposals.")

        model = train_model(
            model, dataset,
            max_epochs=100, patience=8, lr=1e-4, device=device,
            feedback_refiner=refiner, refiner_steps_per_epoch=5,
        )

        # Save checkpoint
        prior_round = checkpoint.get("round", 0) if CHECKPOINT_PATH.exists() else 0
        round_num = prior_round + 1
        torch.save({
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "round":        round_num,
            "num_verified": num_verified,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved → {CHECKPOINT_PATH}  (round {round_num})")

        print("\n--- Running Active Learning Engine ---")
        engine = SelectionEngine(model, str(UNLABELED_DIR), str(PENDING_DIR), device=device)
        engine.run_selection(top_k=5, thresholds=[0.4, 0.5, 0.6], n_passes=10)

        print("Active learning inference complete. Please run the UI to verify uncertain proposals.")
        return

    # 4. If pending review has data, launch UI
    if num_pending > 0:
        print("\nOpening Oracle UI. Please verify pending proposals in the Desktop window.")
        ui = OracleUI(
            review_dir=str(PENDING_DIR),
            verified_dir=str(VERIFIED_DIR),
            rejected_dir=str(REJECTED_DIR)
        )
        ui.launch()

def run_size_test(args):
    """
    Visualizes min and max feature sizes on a full image canvas and a single patch,
    side by side, so you can verify area parameters are correct.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    IMAGE_SIZE = 1024
    PATCH_SIZE = 128
    total_pixels = IMAGE_SIZE * IMAGE_SIZE

    min_area = int(args.min_size * total_pixels)
    max_area = int(args.max_size * total_pixels)

    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = int(np.sqrt(max_area / np.pi))

    print(f"Image size:   {IMAGE_SIZE}x{IMAGE_SIZE} ({total_pixels:,} px)")
    print(f"Min area:     {min_area} px  (r≈{min_radius} px)")
    print(f"Max area:     {max_area} px  (r≈{max_radius} px)")

    def draw_canvas(size, title):
        canvas = np.full((size, size), 40, dtype=np.uint8)  # dark gray background
        cx, cy = size // 2, size // 2
        # Draw max circle (red) then min circle (blue) on top
        cv2.circle(canvas, (cx, cy), max_radius, 220, -1)
        cv2.circle(canvas, (cx, cy), min_radius, 120, -1)
        return canvas

    full_canvas = draw_canvas(IMAGE_SIZE, "Full image")
    patch_canvas = draw_canvas(PATCH_SIZE, "Patch")

    # Try to load a real sample patch from the pipeline data
    sample_patch = None
    sample_patch_path = None
    for search_dir in [
        Path("active_learning_data/pending_review/images"),
        Path("active_learning_data/unlabeled"),
        Path("active_learning_data/verified_dataset/images"),
    ]:
        candidates = sorted(search_dir.glob("*.png")) + sorted(search_dir.glob("*.tif"))
        if candidates:
            sample_patch_path = candidates[0]
            sample_patch = cv2.imread(str(sample_patch_path), cv2.IMREAD_GRAYSCALE)
            break

    ncols = 3 if sample_patch is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    axes[0].imshow(full_canvas, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Full image ({IMAGE_SIZE}x{IMAGE_SIZE})")
    axes[0].axis("off")

    axes[1].imshow(patch_canvas, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Patch ({PATCH_SIZE}x{PATCH_SIZE})")
    axes[1].axis("off")

    if sample_patch is not None:
        axes[2].imshow(sample_patch, cmap="gray")
        axes[2].set_title(f"Sample patch: {sample_patch_path.name}")
        axes[2].axis("off")
        print(f"Sample patch: {sample_patch_path}")

    legend = [
        mpatches.Patch(color="white", label=f"Min: {min_area} px (r≈{min_radius}) = {args.min_size*100:.3f}% of image"),
        mpatches.Patch(color="lightgray", label=f"Max: {max_area} px (r≈{max_radius}) = {args.max_size*100:.3f}% of image"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, framealpha=0.8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = "size_test.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset",       action="store_true", help="Delete all pipeline data (unlabeled, pending_review, verified_dataset, rejected).")
    parser.add_argument("--test",        action="store_true", help="Visualize min/max feature sizes on full image and patch canvas.")
    parser.add_argument("--train",       action="store_true", help="Move remaining pending images back to unlabeled and proceed to training now.")
    # Detection parameters (used by bootstrap and --test)
    parser.add_argument("--min-size",    type=float, default=DEFAULT_MIN_SIZE,    help="Min blob area as fraction of image (e.g. 0.00003).")
    parser.add_argument("--max-size",    type=float, default=DEFAULT_MAX_SIZE,    help="Max blob area as fraction of image (e.g. 0.00040).")
    parser.add_argument("--contrast",    type=int,   default=DEFAULT_CONTRAST,    help="Min local contrast (0–255). Raise to reduce false positives.")
    parser.add_argument("--scale",       type=float, default=DEFAULT_SCALE,       help="Background Gaussian sigma (px). Should exceed blob radius.")
    parser.add_argument("--pre-smooth",  type=float, default=DEFAULT_PRE_SMOOTH,  help="Pre-smoothing Gaussian sigma (px) to suppress sensor noise.")
    parser.add_argument("--circularity", type=float, default=DEFAULT_CIRCULARITY, help="Min circularity 0–1 (0 = disabled, 1 = perfect circle).")
    parser.add_argument("--dilation",    type=int,   default=DEFAULT_DILATION,    help="Expand each detected contour outward by this many pixels (0 = off).")
    args = parser.parse_args()

    if args.reset:
        BASE_DIR = Path("active_learning_data")
        dirs_to_clear = [
            BASE_DIR / "unlabeled",
            BASE_DIR / "pending_review",
            BASE_DIR / "verified_dataset",
            BASE_DIR / "rejected",
            BASE_DIR / "checkpoints",
        ]
        for d in dirs_to_clear:
            if d.exists():
                shutil.rmtree(d)
                print(f"Deleted: {d}")
        print("Reset complete. raw_data has been preserved.")
    elif args.test:
        run_size_test(args)
    else:
        main(args)
