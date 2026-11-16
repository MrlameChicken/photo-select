"""Fast, deterministic image metrics. Scores are heuristics, not guarantees."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def register_heic() -> bool:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def image_paths(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    return sorted(p.resolve() for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_image(path: Path, max_side: int = 1600) -> tuple[Image.Image, tuple[int, int], str]:
    """Decode image; apply EXIF rotation; return resized RGB copy and original dimensions."""
    with Image.open(path) as original:
        exif = original.getexif()
        taken = str(exif.get(36867) or exif.get(36868) or "")
        if not taken:
            taken = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y:%m:%d %H:%M:%S")
        image = ImageOps.exif_transpose(original).convert("RGB")
        dimensions = image.size
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image, dimensions, taken


def dhash(image: Image.Image) -> str:
    """64-bit difference hash for rough near-duplicate filtering."""
    arr = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def metrics(image: Image.Image, width: int, height: int) -> dict[str, float]:
    # Resize to comparable scale so sharpness is more meaningful across photos.
    gray_image = ImageOps.grayscale(image)
    gray_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    gray = np.asarray(gray_image, dtype=np.float32)
    if min(gray.shape) < 3:
        variance = 0.0
    else:
        center = gray[1:-1, 1:-1]
        lap = 4 * center - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        variance = float(np.var(lap))
    sharpness = float(np.clip(np.log1p(variance) / np.log1p(500), 0, 1))
    brightness = float(np.mean(gray) / 255)
    exposure = max(0.0, 1 - abs(brightness - 0.5) * 1.65)
    contrast = float(np.clip(np.std(gray) / 70, 0, 1))
    resolution = float(np.clip(min(width, height) / 1600, 0, 1))
    quality = round(100 * (0.40 * sharpness + 0.25 * exposure + 0.15 * contrast + 0.20 * resolution), 2)
    return {
        "sharpness": round(sharpness, 4),
        "exposure": round(exposure, 4),
        "quality": quality,
    }


def analyze(path: Path) -> tuple[dict, Image.Image]:
    stat = path.stat()
    image, (width, height), taken = load_image(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "width": width,
        "height": height,
        "taken_at": taken,
        "dhash": dhash(image),
        **metrics(image, width, height),
    }, image
