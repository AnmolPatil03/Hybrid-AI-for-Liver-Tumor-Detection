"""
Inference for the ConvNeXt-Tiny U-Net segmentation model.

The model is trained on 2.5D input (previous, current and next CT slice as
3 channels). If only one slice is uploaded, it is repeated for all three
channels.

Returns the tumor mask, location (centroid and bounding box), size,
grayscale intensity statistics, an overlay image and a Seg-Grad-CAM heatmap.
"""

import os
import io
import base64
import torch
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.functional as TF

from src.segmentation_model_convnext import ConvNeXtTinyUNet
from src.gradcam import SegGradCAM

SEG_IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationInference:
    """Loads the segmentation model and runs predictions on uploaded CT slices."""

    def __init__(self):
        self.model = None
        self.device = None
        self.gradcam = None
        self.is_loaded = False

    def load_model(self, model_path='outputs/models/best_segmentation_convnext_model.pth'):
        """Load the trained segmentation model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Segmentation model not found at '{model_path}'. "
                f"Train it first: python train_segmentation_convnext.py"
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading segmentation model on device: {self.device}")

        self.model = ConvNeXtTinyUNet(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.gradcam = SegGradCAM(self.model)

        self.is_loaded = True
        print(f"[INFO] Segmentation model loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    def _load_slice(self, image_bytes):
        """Bytes -> grayscale PIL image, resized to the model's working resolution."""
        img = PILImage.open(io.BytesIO(image_bytes))
        if img.mode != 'L':
            img = img.convert('L')
        return img.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE), PILImage.BILINEAR)

    def preprocess_stack(self, center_bytes, prev_bytes=None, next_bytes=None):
        """
        Build the (1, 3, H, W) input the same way as LiTSSeg25DDataset.
        Returns (input_tensor, original_center_image).
        """
        center_img = self._load_slice(center_bytes)
        prev_img = self._load_slice(prev_bytes) if prev_bytes else center_img
        next_img = self._load_slice(next_bytes) if next_bytes else center_img

        original_image = PILImage.open(io.BytesIO(center_bytes))
        if original_image.mode not in ('L', 'RGB'):
            original_image = original_image.convert('L')
        original_image = original_image.copy()

        channels = [TF.to_tensor(im) for im in (prev_img, center_img, next_img)]  # each (1,H,W)
        image = torch.cat(channels, dim=0)  # (3, H, W)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image.unsqueeze(0), original_image

    def predict_mask(self, center_bytes, prev_bytes=None, next_bytes=None, threshold=0.5):
        """
        Predict the tumor segmentation mask.

        Args:
            center_bytes: Raw bytes of the CT slice being analyzed.
            prev_bytes/next_bytes: Adjacent slices (optional, defaults to the centre slice).
            threshold: Probability threshold for binarizing the mask.

        Returns:
            Dict with mask, location, size, and visualization data.
        """
        if not self.is_loaded:
            raise RuntimeError("Segmentation model not loaded. Call load_model() first.")

        input_tensor, original_image = self.preprocess_stack(center_bytes, prev_bytes, next_bytes)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # (H, W)

        # Binarize mask
        binary_mask = (prob_mask > threshold).astype(np.uint8)

        # Compute tumor metrics
        tumor_info = self._compute_tumor_metrics(binary_mask, original_image.size)

        # Real grayscale intensity stats inside the mask (not calibrated HU).
        intensity = self._compute_intensity_stats(original_image, binary_mask)

        # Generate overlay visualization
        overlay_b64 = self._generate_overlay(original_image, binary_mask, prob_mask)

        # Generate mask image (black and white)
        mask_b64 = self._generate_mask_image(binary_mask)

        # Grad-CAM needs gradients, so it runs outside the no_grad block.
        gradcam_b64 = None
        try:
            overlay, _ = self.gradcam.generate_overlay(
                input_tensor, original_image, device=self.device, image_size=SEG_IMAGE_SIZE,
            )
            gradcam_image = PILImage.fromarray(overlay)
            buffer = io.BytesIO()
            gradcam_image.save(buffer, format='PNG')
            gradcam_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"[WARNING] Seg-Grad-CAM generation failed: {e}")

        return {
            'has_tumor': tumor_info['has_tumor'],
            'tumor_area_pixels': tumor_info['area_pixels'],
            'tumor_area_percentage': tumor_info['area_percentage'],
            'tumor_location': tumor_info['location'],
            'tumor_bbox': tumor_info['bbox'],
            'tumor_intensity': intensity,
            'segmentation_overlay': overlay_b64,
            'segmentation_mask': mask_b64,
            'gradcam_image': gradcam_b64,
        }

    def _compute_intensity_stats(self, original_image, binary_mask):
        """
        Grayscale intensity stats (0-255) inside the tumor mask. These are PNG
        pixel values, not Hounsfield Units.
        """
        if binary_mask.sum() == 0:
            return None
        gray = original_image.convert('L').resize((binary_mask.shape[1], binary_mask.shape[0]))
        arr = np.array(gray)
        vals = arr[binary_mask == 1]
        if vals.size == 0:
            return None
        return {
            'min': int(vals.min()),
            'mean': round(float(vals.mean()), 1),
            'max': int(vals.max()),
        }

    def _compute_tumor_metrics(self, binary_mask, original_size):
        """
        Compute tumor location and size from the binary mask.

        Args:
            binary_mask: Binary numpy array (H, W) with 1=tumor, 0=background.
            original_size: (width, height) of the original image.

        Returns:
            Dict with tumor metrics.
        """
        mask_h, mask_w = binary_mask.shape
        tumor_pixels = np.where(binary_mask == 1)

        if len(tumor_pixels[0]) == 0:
            return {
                'has_tumor': False,
                'area_pixels': 0,
                'area_percentage': 0.0,
                'location': None,
                'bbox': None,
            }

        # Area
        area_pixels = int(len(tumor_pixels[0]))
        total_pixels = mask_h * mask_w
        area_percentage = round((area_pixels / total_pixels) * 100, 2)

        # Centroid (in mask coordinates, then scale to original)
        centroid_y = float(np.mean(tumor_pixels[0]))
        centroid_x = float(np.mean(tumor_pixels[1]))

        # Scale to original image coordinates
        orig_w, orig_h = original_size
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

        location = {
            'centroid_x': round(centroid_x * scale_x, 1),
            'centroid_y': round(centroid_y * scale_y, 1),
            'centroid_x_normalized': round(centroid_x / mask_w, 3),
            'centroid_y_normalized': round(centroid_y / mask_h, 3),
        }

        # Bounding box
        y_min, y_max = int(tumor_pixels[0].min()), int(tumor_pixels[0].max())
        x_min, x_max = int(tumor_pixels[1].min()), int(tumor_pixels[1].max())

        bbox = {
            'x_min': round(x_min * scale_x, 1),
            'y_min': round(y_min * scale_y, 1),
            'x_max': round(x_max * scale_x, 1),
            'y_max': round(y_max * scale_y, 1),
            'width': round((x_max - x_min) * scale_x, 1),
            'height': round((y_max - y_min) * scale_y, 1),
        }

        return {
            'has_tumor': True,
            'area_pixels': area_pixels,
            'area_percentage': area_percentage,
            'location': location,
            'bbox': bbox,
        }

    def _generate_overlay(self, original_image, binary_mask, prob_mask):
        """Generate a colored overlay of the segmentation on the original image."""
        # Resize original to mask size
        orig_resized = original_image.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE))
        if orig_resized.mode == 'L':
            orig_resized = orig_resized.convert('RGB')
        orig_array = np.array(orig_resized)

        # Create red overlay for tumor regions
        overlay = orig_array.copy()
        tumor_mask_3d = np.stack([binary_mask] * 3, axis=-1).astype(bool)

        # Red tint on tumor regions
        overlay[tumor_mask_3d[:, :, 0], 0] = np.clip(
            overlay[tumor_mask_3d[:, :, 0], 0].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        overlay[tumor_mask_3d[:, :, 1], 1] = np.clip(
            overlay[tumor_mask_3d[:, :, 1], 1].astype(int) - 50, 0, 255
        ).astype(np.uint8)
        overlay[tumor_mask_3d[:, :, 2], 2] = np.clip(
            overlay[tumor_mask_3d[:, :, 2], 2].astype(int) - 50, 0, 255
        ).astype(np.uint8)

        # Blend
        alpha = 0.6
        result = (alpha * orig_array + (1 - alpha) * overlay).astype(np.uint8)

        # Encode to base64
        result_image = PILImage.fromarray(result)
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_mask_image(self, binary_mask):
        """Generate a base64-encoded mask image."""
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        mask_image = PILImage.fromarray(mask_uint8, mode='L')
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Global segmentation inference engine
segmentation_engine = SegmentationInference()
