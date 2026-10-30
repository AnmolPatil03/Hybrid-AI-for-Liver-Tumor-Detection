"""
ConvNeXt-Tiny encoder + U-Net decoder for liver tumor segmentation.

Uses segmentation_models_pytorch (`tu-convnext_tiny`) when installed and
falls back to a custom decoder on a timm ConvNeXt-Tiny backbone otherwise.

Input:  (B, 3, 256, 256)  2.5D stack of neighbouring CT slices
Output: (B, 1, 256, 256)  tumor logits
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False

import timm


class ConvNeXtTinyUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ConvNeXt-Tiny encoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        if _HAS_SMP:
            self._impl = "smp"
            self.model = smp.Unet(
                encoder_name="tu-convnext_tiny",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=1,
                activation=None,
            )
        else:
            self._impl = "custom"
            self.model = _CustomConvNeXtUNet(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)


class _DecoderBlock(nn.Module):
    """Upsample, concatenate skip connection, double conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class _CustomConvNeXtUNet(nn.Module):
    """
    U-Net decoder over timm's ConvNeXt-Tiny feature stages
    (strides 4/8/16/32, channels 96/192/384/768).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        chs = self.encoder.feature_info.channels()  # [96, 192, 384, 768]

        self.decoder4 = _DecoderBlock(chs[3], chs[2], 256)  # H/32 -> H/16
        self.decoder3 = _DecoderBlock(256, chs[1], 128)      # H/16 -> H/8
        self.decoder2 = _DecoderBlock(128, chs[0], 64)       # H/8  -> H/4

        # The ConvNeXt stem downsamples by 4, so two more upsamples are needed.
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # H/4 -> H/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),   # H/2 -> H
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)   # H/4, H/8, H/16, H/32
        d4 = self.decoder4(f3, f2)         # H/16
        d3 = self.decoder3(d4, f1)         # H/8
        d2 = self.decoder2(d3, f0)         # H/4
        out = self.final_upsample(d2)      # H
        out = self.final_conv(out)
        return out


class DiceLoss(nn.Module):
    """Soft Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky loss: weights false positives by alpha and false negatives by
    beta (alpha = beta = 0.5 gives Dice). beta > alpha favours recall.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1.0 - target_flat)).sum()
        fn = ((1.0 - pred_flat) * target_flat).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, 2019): Tversky loss ** (1 / gamma)."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0, smooth=1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, pred, target):
        tversky_loss = self.tversky(pred, target)
        return torch.pow(tversky_loss, 1.0 / self.gamma)


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCE (with optional pos_weight for class imbalance) and
    Tversky / Focal Tversky loss.
    """

    def __init__(self, bce_weight=0.5, region_weight=0.5, alpha=0.3, beta=0.7,
                 pos_weight=None, use_focal=False, gamma=4.0 / 3.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.region = (
            FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma) if use_focal
            else TverskyLoss(alpha=alpha, beta=beta)
        )
        self.bce_weight = bce_weight
        self.region_weight = region_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.region_weight * self.region(pred, target)


def create_segmentation_model(pretrained=True):
    model = ConvNeXtTinyUNet(pretrained=pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] ConvNeXt-Tiny U-Net segmentation model created (impl={model._impl})")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    return model
