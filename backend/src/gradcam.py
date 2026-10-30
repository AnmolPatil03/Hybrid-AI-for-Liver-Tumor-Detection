"""
Seg-Grad-CAM for the ConvNeXt-Tiny U-Net.

A segmentation model has no single class logit, so the gradient is taken
from the sum of predicted tumor probabilities over the image. The weighted
activation map of an encoder stage then shows which regions drove the
"how much tumor" score.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    Vinogradova et al., "Towards Interpretable Semantic Segmentation via
        Gradient-Weighted Class Activation Mapping"
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SegGradCAM:
    """
    Grad-CAM for ConvNeXtTinyUNet (smp or custom implementation).

    The encoder and decoder are called directly so the hooked feature map
    can use retain_grad(); the encoder returns a list of tensors, which
    module backward hooks don't handle reliably.

    features[-3] (32x32 at 256x256 input) is used instead of the deepest
    8x8 stage, because many LiTS tumors are smaller than one 32x32-pixel
    cell of the deepest map and cannot be localised from it.
    """

    HOOK_STAGE = -3

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.inner = model.model
        self.impl = model._impl  # "smp" or "custom"

    def _forward_with_hooked_feature(self, x):
        """Run encoder -> decoder ourselves; return (logits, hooked_feature)."""
        features = self.inner.encoder(x)   # List[Tensor], shallow -> deep
        hooked = features[self.HOOK_STAGE]
        hooked.retain_grad()

        if self.impl == "smp":
            decoder_output = self.inner.decoder(features)
            logits = self.inner.segmentation_head(decoder_output)
        else:
            f0, f1, f2, f3 = features
            d4 = self.inner.decoder4(f3, f2)
            d3 = self.inner.decoder3(d4, f1)
            d2 = self.inner.decoder2(d3, f0)
            out = self.inner.final_upsample(d2)
            logits = self.inner.final_conv(out)

        return logits, hooked

    def generate_heatmap(self, input_tensor, device='cpu', image_size=256):
        """
        Generate a Seg-Grad-CAM heatmap for the given input image tensor.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            device: torch.device or device string.
            image_size: Output heatmap resolution.

        Returns:
            heatmap: NumPy array of shape (image_size, image_size), values in [0, 1].
        """
        input_tensor = input_tensor.to(device)
        self.model = self.model.to(device)

        logits, hooked = self._forward_with_hooked_feature(input_tensor)

        self.model.zero_grad()
        # Sum (not mean) so the gradient stays usable for small tumors.
        target = torch.sigmoid(logits).sum()
        target.backward()

        gradients = hooked.grad             # (1, C, h, w)
        activations = hooked.detach()       # (1, C, h, w)

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image_size, image_size), mode='bilinear', align_corners=False)

        # The raw CAM is diffuse because the target is a whole-image sum;
        # weighting it by the predicted probability keeps it on the tumor.
        with torch.no_grad():
            pred_prob = torch.sigmoid(logits)
            pred_prob = F.interpolate(pred_prob, size=(image_size, image_size),
                                       mode='bilinear', align_corners=False)
        cam = cam * pred_prob

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_overlay(self, input_tensor, original_image, device='cpu', alpha=0.5, image_size=256):
        """
        Generate a Seg-Grad-CAM heatmap overlay on the original image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            original_image: Original PIL image (before preprocessing).
            device: torch.device or device string.
            alpha: Blend weight of the heatmap in the overlay.
            image_size: resolution to render at.

        Returns:
            overlay: NumPy array (H, W, 3) with heatmap overlay on the original image.
            heatmap: Raw heatmap array (image_size, image_size).
        """
        heatmap = self.generate_heatmap(input_tensor, device, image_size=image_size)

        original_resized = original_image.resize((image_size, image_size))
        original_array = np.array(original_resized)
        if original_array.ndim == 2:
            original_array = np.stack([original_array] * 3, axis=-1)
        if original_array.shape[-1] == 4:
            original_array = original_array[..., :3]

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_array)

        return overlay, heatmap

