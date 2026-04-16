import os
import cv2
import glob
import torch
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple
from model import ResAttentionUNet
from patch_utils import extract_patches

class SelectionEngine:
    """
    Active Learning selection engine.
    Runs the unlabeled dataset through the model using MC Dropout to get uncertainty scores.
    Ranks images by uncertainty, and selects the most confusing ones for the oracle.
    Generates multiple boundary proposals for high uncertainty images.
    """
    def __init__(self, model: ResAttentionUNet, unlabeled_dir: str, review_dir: str, device: str = 'cuda'):
        self.model = model
        self.unlabeled_dir = Path(unlabeled_dir)
        self.review_dir = Path(review_dir)
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model.to(self.device)
        
        # Ensure review directory structure exists
        self.review_dir.mkdir(parents=True, exist_ok=True)
        (self.review_dir / "images").mkdir(exist_ok=True)
        (self.review_dir / "masks").mkdir(exist_ok=True)
        (self.review_dir / "overlays").mkdir(exist_ok=True)

    def run_selection(self, top_k: int = 10, thresholds: List[float] = [0.4, 0.5, 0.6], n_passes: int = 10):
        """
        Rank unlabeled images and move the top_k most uncertain images to the review queue.
        Calculates multiple boundary proposals for these selected images.
        """
        image_paths = sorted(glob.glob(str(self.unlabeled_dir / "*.*")))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Pre-process: chunk any large images into patches
        PATCH_SIZE = 128
        new_patches_created = False
        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None: continue
            
            if img_gray.shape[0] > PATCH_SIZE or img_gray.shape[1] > PATCH_SIZE:
                patches = extract_patches(img_gray, patch_size=(PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))
                stem = img_path.stem
                for patch_img, x, y in patches:
                    cv2.imwrite(str(self.unlabeled_dir / f"{stem}_patch_x{x}_y{y}.png"), patch_img)
                os.remove(img_path)
                new_patches_created = True

        # Refresh list if we created new patches
        if new_patches_created:
            image_paths = sorted(glob.glob(str(self.unlabeled_dir / "*.*")))
            image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not image_paths:
            print("No unlabeled images found.")
            return

        print(f"Running uncertainty estimation on {len(image_paths)} images...")
        
        uncertainty_scores = []
        # Store predictive means to avoid recomputing for top K
        predictive_means = {}

        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue

            # Preprocess for model: (1, 1, H, W)
            # Assuming basic normalization [0, 1]
            input_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(self.device)

            # Get mean probability and uncertainty score
            mean_probs, uncertainty = self.model.inference_with_uncertainty(input_tensor, n_passes=n_passes)
            
            uncertainty_scores.append((uncertainty, img_path))
            predictive_means[str(img_path)] = mean_probs

        # Rank (descending order of uncertainty)
        uncertainty_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_items = uncertainty_scores[:top_k]

        print(f"Selecting top {len(top_k_items)} uncertain images for oracle review.")

        for rank, (score, img_path) in enumerate(top_k_items):
            print(f"Rank {rank+1}: {img_path.name} (Uncertainty: {score:.4f})")
            mean_probs = predictive_means[str(img_path)]
            self._generate_proposals_and_move(img_path, mean_probs, thresholds)

    def _generate_proposals_and_move(self, img_path: Path, mean_probs: np.ndarray, thresholds: List[float]):
        """
        Generates boundary overlays for different thresholds to give oracle options.
        Moves the original image and proposals to the pending_review directory.
        Removes the image from the unlabeled directory.
        """
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        stem = img_path.stem

        # Save the original image to review dir
        cv2.imwrite(str(self.review_dir / "images" / f"{stem}.png"), img_gray)

        for thresh in thresholds:
            # Binarize using threshold
            binary_mask = (mean_probs >= thresh).astype(np.uint8) * 255
            
            # Extract contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create overlay
            overlay_img = img_bgr.copy()
            cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)  # Red contours

            # Save proposals with threshold identifier
            thresh_str = str(thresh).replace(".", "")
            cv2.imwrite(str(self.review_dir / "masks" / f"{stem}_thresh_{thresh_str}.png"), binary_mask)
            cv2.imwrite(str(self.review_dir / "overlays" / f"{stem}_thresh_{thresh_str}_overlay.png"), overlay_img)

        # Move the original image from unlabeled so it isn't processed again
        os.remove(img_path)
