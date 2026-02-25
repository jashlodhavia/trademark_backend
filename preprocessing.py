"""
preprocessing.py
────────────────
Background removal (rembg / U2-Net), foreground tight-cropping,
and neutral-gray canvas compositing.

All visual embeddings (DINO, VGG, color) should operate on the
output of preprocess_image() to eliminate background bias.
"""

import numpy as np
from PIL import Image
from rembg import remove, new_session


GRAY_FILL = (128, 128, 128)

_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2netp")
    return _rembg_session


def extract_foreground(image: Image.Image) -> Image.Image:
    """
    Remove background via U2-Net-lite (u2netp), tight-crop to the
    foreground bounding box, and composite onto a neutral-gray canvas.

    Gray (128,128,128) maps to ~0 after ImageNet normalization, so
    background pixels contribute minimal activation in DINO/VGG.
    """
    fg_rgba = remove(image.convert("RGB"), session=_get_rembg_session())
    alpha = np.array(fg_rgba.split()[-1])

    coords = np.argwhere(alpha > 128)
    if coords.size == 0:
        return image.convert("RGB")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    fg_crop = fg_rgba.crop((x0, y0, x1 + 1, y1 + 1))
    canvas = Image.new("RGB", fg_crop.size, GRAY_FILL)
    canvas.paste(fg_crop, mask=fg_crop.split()[-1])
    return canvas


def preprocess_image(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Full preprocessing pipeline:
      raw image → background removal → tight crop → square pad → resize
    """
    fg = extract_foreground(image)
    return pad_and_resize(fg, target_size)


def pad_and_resize(image: Image.Image, target_size: int = 224) -> Image.Image:
    """Square-pad and resize only (no background removal). Fast."""
    w, h = image.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), GRAY_FILL)
    padded.paste(image, ((size - w) // 2, (size - h) // 2))
    return padded.resize((target_size, target_size), Image.LANCZOS)
