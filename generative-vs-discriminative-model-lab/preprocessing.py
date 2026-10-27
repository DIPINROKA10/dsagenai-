"""
preprocessing.py - MNIST Digit Image Preprocessing Pipeline

Converts raw canvas strokes or uploaded user images into normalized 28x28 grayscale
tensors matching the standard MNIST format (white digit on black background, centered by center of mass).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
import torch
import torchvision.transforms as transforms


def is_blank_image(image_np: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Check if the input grayscale image is essentially blank.
    threshold: maximum pixel value to consider as blank (0-255 scale).
    """
    if image_np is None or image_np.size == 0:
        return True
    return bool(np.max(image_np) < threshold)


def preprocess_image(image_input, invert_if_dark_on_light: bool = True) -> torch.Tensor:
    """
    Preprocess a PIL Image or numpy array into a 1x1x28x28 PyTorch tensor matching MNIST format.

    Pipeline:
    1. Convert to grayscale (L mode).
    2. Auto-invert if digit is dark on white background (MNIST requires white digit on black background).
    3. Find bounding box around the active digit pixels.
    4. Crop to bounding box and resize to fit within a 20x20 box preserving aspect ratio.
    5. Center the 20x20 digit inside a 28x28 black canvas using the center of mass.
    6. Normalize pixel values to [0.0, 1.0] and convert to PyTorch FloatTensor of shape (1, 1, 28, 28).

    Args:
        image_input: PIL.Image.Image or np.ndarray (RGB, RGBA, or Grayscale).
        invert_if_dark_on_light: Automatically invert if background is light.

    Returns:
        torch.Tensor of shape (1, 1, 28, 28) with values in [0, 1].
        Raises ValueError if image is empty or blank.
    """
    # 1. Convert input to PIL Image
    if isinstance(image_input, np.ndarray):
        # Handle RGBA canvas arrays (e.g., from streamlit-drawable-canvas)
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            # Check alpha channel: if strokes are in alpha or RGB
            alpha = image_input[:, :, 3]
            rgb = image_input[:, :, :3]
            if np.max(alpha) > 0 and np.mean(rgb) < 10:
                # White stroke on transparent canvas
                img = Image.fromarray(alpha).convert("L")
            else:
                # Standard RGB or RGBA on white canvas
                img = Image.fromarray(image_input).convert("L")
        elif image_input.ndim == 3:
            img = Image.fromarray(image_input).convert("L")
        else:
            img = Image.fromarray(image_input.astype(np.uint8)).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    img_np = np.array(img, dtype=np.float32)

    # 2. Check for empty canvas
    if img_np.size == 0 or np.ptp(img_np) < 5:
        raise ValueError("Image appears to be empty. Please draw a digit or upload an image.")

    # 3. Check background polarity: MNIST needs white digit (255) on black background (0)
    # Estimate background from the 4 corners
    h, w = img_np.shape
    corner_pixels = np.array([img_np[0, 0], img_np[0, -1], img_np[-1, 0], img_np[-1, -1]])
    corner_mean = float(np.mean(corner_pixels))

    if invert_if_dark_on_light and corner_mean > 127.0:
        # Background is white/light, so invert
        img_np = 255.0 - img_np

    # Threshold noise: zero out weak values
    img_np[img_np < 25.0] = 0.0

    if np.max(img_np) < 15.0:
        raise ValueError("No distinct stroke detected after contrast normalization.")

    # 4. Crop to bounding box of active pixels
    active_coords = np.argwhere(img_np > 30.0)
    if len(active_coords) == 0:
        raise ValueError("Could not locate digit stroke in the image.")

    y_min, x_min = active_coords.min(axis=0)
    y_max, x_max = active_coords.max(axis=0)

    # Add small padding around bounding box
    pad = 2
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(h - 1, y_max + pad)
    x_max = min(w - 1, x_max + pad)

    cropped_np = img_np[y_min : y_max + 1, x_min : x_max + 1]
    cropped_img = Image.fromarray(cropped_np.astype(np.uint8))

    # 5. Fit inside 20x20 box preserving aspect ratio
    box_size = 20
    crop_w, crop_h = cropped_img.size

    if crop_w == 0 or crop_h == 0:
        raise ValueError("Invalid cropped stroke dimensions.")

    scale = box_size / max(crop_w, crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_np = np.array(resized_img, dtype=np.float32)

    # 6. Place on a 28x28 blank canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)

    # Initial placement: center geometrically
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_np

    # 7. Shift to center of mass (standard MNIST centering convention)
    cy, cx = center_of_mass(canvas_28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))

        # Clamp shift to avoid clipping off canvas
        shift_y = max(-4, min(4, shift_y))
        shift_x = max(-4, min(4, shift_x))

        if shift_y != 0 or shift_x != 0:
            shifted = np.zeros_like(canvas_28)
            src_y1 = max(0, -shift_y)
            src_y2 = min(28, 28 - shift_y)
            src_x1 = max(0, -shift_x)
            src_x2 = min(28, 28 - shift_x)

            dst_y1 = max(0, shift_y)
            dst_y2 = min(28, 28 + shift_y)
            dst_x1 = max(0, shift_x)
            dst_x2 = min(28, 28 + shift_x)

            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = canvas_28[src_y1:src_y2, src_x1:src_x2]
            canvas_28 = shifted

    # 8. Normalize to [0.0, 1.0]
    canvas_28 = np.clip(canvas_28 / 255.0, 0.0, 1.0)

    # 9. Convert to PyTorch Tensor (1, 1, 28, 28)
    tensor = torch.from_numpy(canvas_28).float().unsqueeze(0).unsqueeze(0)
    return tensor
