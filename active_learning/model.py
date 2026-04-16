import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    Initialized with Kaiming Normal to handle ReLU activations.
    Keeps channel count constant (in == out).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al. 2018).
    Learns to focus on relevant spatial locations in the skip connection.

    Args:
        F_g:   channels in the gate signal  (from decoder, coarser scale)
        F_l:   channels in the skip signal  (from encoder, finer scale)
        F_int: intermediate feature channels (typically min(F_g, F_l) // 2)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: gate signal   (B, F_g, H', W')  — upsampled decoder feature
        x: skip signal   (B, F_l, H,  W )  — encoder skip connection
        Returns x weighted by learned attention coefficients.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))   # (B, 1, H, W)
        return x * alpha


class EncoderBlock(nn.Module):
    """
    MaxPool → channel-transition conv → ResidualBlock → MC Dropout.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = ResidualBlock(out_ch)
        self.drop = nn.Dropout2d(p=dropout_rate)
        self._init_conv()

    def _init_conv(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.res(x)
        return self.drop(x)


class DecoderBlock(nn.Module):
    """
    ConvTranspose2d → AttentionGate on skip → concat → conv → ResidualBlock → MC Dropout.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_rate: float = 0.3):
        super().__init__()
        up_ch = in_ch // 2
        self.up = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(F_g=up_ch, F_l=skip_ch, F_int=max(up_ch // 2, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(up_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = ResidualBlock(out_ch)
        self.drop = nn.Dropout2d(p=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for ConvTranspose2d and the channel-fusion conv
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip spatial dims if input was odd-sized
        dY = skip.size(2) - x.size(2)
        dX = skip.size(3) - x.size(3)
        if dY > 0 or dX > 0:
            x = F.pad(x, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        skip = self.attn(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return self.drop(x)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class ResAttentionUNet(nn.Module):
    """
    Residual Attention U-Net for small-feature segmentation in TEM images.

    Architecture highlights:
    - Residual blocks (Kaiming Normal init) for stable gradient flow through deep U
    - Attention gates in every decoder stage to focus on dislocation contrast
    - Orthogonal init on all non-residual convolutions to preserve signal energy
    - Pessimistic output bias (-2.0) to handle extreme class imbalance at init
    - MC Dropout throughout for uncertainty estimation (active learning)

    Input:  (B, 1, H, W)  — normalised single-channel TEM crop
    Output: (B, 1, H, W)  — sigmoid probability map (raw logits in forward())
    """
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        dropout_rate: float = 0.3,
        final_bias: float = -2.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ---- Encoder ----
        # Stem: channel-lifting conv + residual refinement (no pooling)
        self.stem_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem_res = ResidualBlock(64)

        self.enc1 = EncoderBlock(64,  128, dropout_rate)   # /2
        self.enc2 = EncoderBlock(128, 256, dropout_rate)   # /4
        self.enc3 = EncoderBlock(256, 512, dropout_rate)   # /8

        # Bottleneck (deepest encoding, no skip)
        self.bottleneck = EncoderBlock(512, 512, dropout_rate)  # /16

        # ---- Decoder ----
        # dec_i takes the output of the stage below + the matching encoder skip
        self.dec3 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256, dropout_rate=dropout_rate)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128, dropout_rate=dropout_rate)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64,  dropout_rate=dropout_rate)
        self.dec0 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=64,  dropout_rate=dropout_rate)

        # ---- Output head ----
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        nn.init.orthogonal_(self.outc.weight)
        nn.init.constant_(self.outc.bias, final_bias)  # pessimistic init for sparse features

        # Apply orthogonal init to stem conv (residual blocks handle themselves)
        for m in self.stem_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.stem_res(self.stem_conv(x))  # 64 ch
        s1 = self.enc1(s0)                     # 128 ch
        s2 = self.enc2(s1)                     # 256 ch
        s3 = self.enc3(s2)                     # 512 ch
        b  = self.bottleneck(s3)               # 512 ch  (deepest)

        # Decoder
        d3 = self.dec3(b,  s3)   # 256 ch
        d2 = self.dec2(d3, s2)   # 128 ch
        d1 = self.dec1(d2, s1)   # 64  ch
        d0 = self.dec0(d1, s0)   # 64  ch

        return self.outc(d0)

    # ------------------------------------------------------------------
    # MC-Dropout helpers (keep interface identical to old MCDropoutUNet)
    # ------------------------------------------------------------------

    def enable_dropout(self):
        """Forces all Dropout layers to remain active during eval (MC Dropout)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout2d, nn.Dropout)):
                m.train()

    @torch.no_grad()
    def inference_with_uncertainty(
        self,
        image_tensor: torch.Tensor,
        n_passes: int = 10,
    ) -> Tuple[np.ndarray, float]:
        """
        Stochastic forward passes → predictive mean and pixel-entropy uncertainty.

        Args:
            image_tensor: (1, 1, H, W) normalised tensor on the correct device.
            n_passes:     Number of MC Dropout samples.

        Returns:
            predictive_mean:  (H, W) numpy array of averaged sigmoid probabilities.
            uncertainty_score: scalar mean pixel entropy across the image.
        """
        self.eval()
        self.enable_dropout()

        preds = []
        for _ in range(n_passes):
            logits = self(image_tensor)
            preds.append(torch.sigmoid(logits))

        preds = torch.stack(preds)                       # (N, 1, 1, H, W)
        mean_probs = torch.mean(preds, dim=0).squeeze().cpu().numpy()

        eps = 1e-7
        entropy = (
            -mean_probs * np.log(mean_probs + eps)
            - (1.0 - mean_probs) * np.log(1.0 - mean_probs + eps)
        )
        uncertainty_score = float(np.mean(entropy))
        return mean_probs, uncertainty_score
