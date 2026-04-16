"""
trainer.py — Loss functions, training loop, and Human-in-the-Loop refinement.

Key components:
  - DiceLoss:           overlap-based loss, good for class-imbalanced masks
  - FocalLoss:          down-weights easy background pixels, forces focus on
                        rare dislocation pixels (mandatory for sparse features)
  - CombinedLoss:       Dice + Focal (weighted sum)
  - FeedbackRefinement: HITL logic — contrastive penalty for rejected proposals
                        and a priority replay buffer for accepted ones
  - train_model():      training loop using CombinedLoss
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss. Works on sigmoid probabilities (not logits)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al. 2017).
    Down-weights confident background predictions so the model is forced
    to pay attention to the rare, hard-to-classify dislocation pixels.

    Args:
        alpha: balancing factor for positive class (default 0.25)
        gamma: focusing parameter — higher = more focus on hard examples (default 2.0)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    Dice + Focal loss.
    Dice provides global shape supervision; Focal handles per-pixel class imbalance.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.w_dice  = dice_weight
        self.w_focal = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, targets) + self.w_focal * self.focal(logits, targets)


# ---------------------------------------------------------------------------
# Priority replay buffer entry
# ---------------------------------------------------------------------------

class _ReplayItem:
    __slots__ = ("img", "mask", "priority")

    def __init__(self, img: torch.Tensor, mask: torch.Tensor, priority: float):
        self.img      = img
        self.mask     = mask
        self.priority = priority


# ---------------------------------------------------------------------------
# Human-in-the-Loop feedback refinement
# ---------------------------------------------------------------------------

class FeedbackRefinement:
    """
    HITL refinement logic.

    Usage after each oracle review round::

        refiner = FeedbackRefinement(model, optimizer, device)

        # For every image the oracle accepted:
        refiner.process_feedback(img_t, mask_t, expert_score=1)

        # For every image the oracle rejected (with the model's predicted mask):
        refiner.process_feedback(img_t, pred_mask_t, expert_score=0)

        # Optional: fine-tune model on accumulated expert-approved examples:
        refiner.finetune_from_buffer(n_steps=20)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        buffer_capacity: int = 200,
        contrastive_lr_scale: float = 0.1,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.device    = torch.device(device)
        self.buffer: deque[_ReplayItem] = deque(maxlen=buffer_capacity)
        self.loss_fn   = CombinedLoss()
        self.contrastive_lr_scale = contrastive_lr_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_feedback(
        self,
        img_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        expert_score: int,
    ) -> None:
        """
        Process a single human judgement.

        Args:
            img_tensor:   (1, 1, H, W) normalised image tensor.
            mask_tensor:  (1, 1, H, W) mask — approved mask if score=1,
                          model's rejected prediction if score=0.
            expert_score: 1 = accurate proposal (accept), 0 = inaccurate (reject).
        """
        if expert_score == 1:
            self._add_to_buffer(img_tensor, mask_tensor, priority=1.0)
        else:
            self._contrastive_step(img_tensor, mask_tensor)

    def add_approved_from_dir(self, verified_dir: str | Path) -> int:
        """
        Scan verified_dataset/ and load all image–mask pairs into the replay
        buffer at high priority.  Returns number of items added.
        """
        verified_dir = Path(verified_dir)
        img_dir  = verified_dir / "images"
        mask_dir = verified_dir / "masks"
        added = 0
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue
            img_t  = _load_as_tensor(img_path)
            mask_t = _load_as_tensor(mask_path)
            self._add_to_buffer(img_t, mask_t, priority=1.0)
            added += 1
        return added

    def apply_contrastive_from_dir(self, rejected_dir: str | Path) -> int:
        """
        Scan rejected/masks/ for rejected proposals and apply one contrastive
        gradient step per image.  Returns number of items processed.
        """
        rejected_dir = Path(rejected_dir)
        masks_dir = rejected_dir / "masks"
        imgs_dir  = rejected_dir / "images"
        if not masks_dir.exists():
            return 0
        processed = 0
        for mask_path in sorted(masks_dir.glob("*.png")):
            img_path = imgs_dir / mask_path.name
            if not img_path.exists():
                continue
            img_t  = _load_as_tensor(img_path)
            mask_t = _load_as_tensor(mask_path)
            self._contrastive_step(img_t, mask_t)
            processed += 1
        return processed

    def finetune_from_buffer(
        self,
        n_steps: int = 20,
        batch_size: int = 4,
    ) -> List[float]:
        """
        Fine-tune the model on expert-approved examples from the replay buffer.
        Items are sampled proportionally to their priority.

        Returns list of per-step losses.
        """
        if not self.buffer:
            return []

        items = list(self.buffer)
        priorities = torch.tensor([it.priority for it in items], dtype=torch.float32)
        probs = priorities / priorities.sum()

        self.model.train()
        losses = []
        for _ in range(n_steps):
            indices = torch.multinomial(probs, min(batch_size, len(items)), replacement=True)
            batch_imgs  = torch.cat([items[i].img  for i in indices], dim=0).to(self.device)
            batch_masks = torch.cat([items[i].mask for i in indices], dim=0).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_imgs)
            loss = self.loss_fn(logits, batch_masks)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_to_buffer(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        priority: float,
    ) -> None:
        self.buffer.append(_ReplayItem(img.cpu(), mask.cpu(), priority))

    def _contrastive_step(
        self,
        img_tensor: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> float:
        """
        One gradient step that penalises the model for the specific spatial
        pattern it proposed in the rejected mask.

        Strategy: train the model toward the *inverse* of the rejected mask
        (i.e. "don't predict foreground where you just predicted foreground").
        The step is scaled down by `contrastive_lr_scale` to avoid over-correction.
        """
        self.model.train()
        img = img_tensor.to(self.device)

        # Inverted soft target: pixels predicted as dislocation should become background
        inverted_target = (1.0 - rejected_mask.float()).to(self.device)
        inverted_target = torch.clamp(inverted_target, 0.0, 1.0)

        # Temporarily scale down the learning rate
        for pg in self.optimizer.param_groups:
            pg['lr'] *= self.contrastive_lr_scale

        self.optimizer.zero_grad()
        logits = self.model(img)
        loss = F.binary_cross_entropy_with_logits(logits, inverted_target)
        loss.backward()
        self.optimizer.step()

        # Restore learning rate
        for pg in self.optimizer.param_groups:
            pg['lr'] /= self.contrastive_lr_scale

        return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class VerifiedDataset(Dataset):
    """Loads verified image / mask pairs from the verified_dataset/ directory."""

    def __init__(self, verified_dir: str | Path):
        self.verified_dir = Path(verified_dir)
        (self.verified_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.verified_dir / "masks").mkdir(parents=True, exist_ok=True)
        self.image_paths = sorted((self.verified_dir / "images").glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path  = self.image_paths[idx]
        mask_path = self.verified_dir / "masks" / img_path.name
        img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img_t  = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return img_t, mask_t


def train_model(
    model: nn.Module,
    dataset: Dataset,
    max_epochs: int = 100,
    patience: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = 'cpu',
    feedback_refiner: Optional[FeedbackRefinement] = None,
    refiner_steps_per_epoch: int = 5,
) -> nn.Module:
    """
    Train (or fine-tune) the model using CombinedLoss (Dice + Focal).
    Stops early if the loss does not improve for `patience` consecutive epochs.

    Args:
        model:                   The segmentation network.
        dataset:                 VerifiedDataset (or any image/mask Dataset).
        max_epochs:              Hard cap on training epochs (default 100).
        patience:                Stop if loss hasn't improved for this many epochs (default 5).
        batch_size:              Mini-batch size.
        lr:                      Learning rate (recommended: 1e-4 or 5e-5).
        device:                  'cuda' or 'cpu'.
        feedback_refiner:        If provided, runs replay-buffer fine-tuning each epoch.
        refiner_steps_per_epoch: Steps drawn from the replay buffer per epoch.
    """
    if len(dataset) == 0:
        print("Verified dataset is empty — nothing to train on yet.")
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion  = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    optimizer  = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    if feedback_refiner is not None:
        feedback_refiner.optimizer = optimizer

    best_loss     = float('inf')
    epochs_no_imp = 0

    print(f"Training on {len(dataset)} verified samples | "
          f"max {max_epochs} epochs | patience {patience} | lr={lr}")

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        replay_info = ""
        if feedback_refiner is not None and refiner_steps_per_epoch > 0:
            step_losses = feedback_refiner.finetune_from_buffer(
                n_steps=refiner_steps_per_epoch, batch_size=batch_size)
            if step_losses:
                replay_info = f" | replay_loss={np.mean(step_losses):.4f}"

        # Early stopping check
        if avg_loss < best_loss:
            best_loss     = avg_loss
            epochs_no_imp = 0
            stop_info     = ""
        else:
            epochs_no_imp += 1
            stop_info     = f" | no improvement {epochs_no_imp}/{patience}"

        print(f"Epoch {epoch+1}/{max_epochs} — loss={avg_loss:.4f}{replay_info}{stop_info}")

        if epochs_no_imp >= patience:
            print(f"Early stopping: loss flat for {patience} epochs.")
            break

    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _load_as_tensor(path: Path) -> torch.Tensor:
    """Load a grayscale image as a (1, 1, H, W) float tensor in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
