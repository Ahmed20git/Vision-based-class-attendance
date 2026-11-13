"""Make simulated test conditions from ordinary photos.

The brief asks for tests at different distances, lighting and poses. This
script takes normal photos and creates versions that imitate those
conditions, so the same people can be tested under each one:

    original    the photo as it is
    far2x       face half the size, as if twice as far away
    far4x       face a quarter of the size, as if four times as far away
    dark        exposure reduced (dim room)
    verydark    exposure strongly reduced, with sensor noise (very dim room)
    bright      overexposed (strong light / backlit camera auto-exposure)
    blur        out of focus or moving
    noise       grainy low-quality camera
    tilt15      head tilted 15 degrees (in-plane rotation only; turning the
                head sideways cannot be simulated, so use real photos for that)

Usage (from the Evaluation folder):
    ../FaceRecognition/.venv/bin/python make_conditions.py photos_folder

Each output is saved in input/ as <original name>_<condition>.jpg, ready for
collect_faces.py. In the report, say that these conditions are simulated.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from eval_common import INPUT_DIR

RNG = np.random.default_rng(0)  # fixed seed so the same files come out every time


def shrink(image, factor):
    """Make faces smaller (further away): scale down, then back up to the
    original size, so the detail of a far-away face is lost."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(image, int(image.mean()))
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def exposure(image, gain, gamma):
    """Change brightness like a camera exposure (gain) with a gamma curve."""
    scaled = np.clip((image.astype(np.float32) / 255.0) * gain, 0, 1) ** gamma
    return (scaled * 255).astype(np.uint8)


def add_noise(image, sigma):
    noisy = image.astype(np.float32) + RNG.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate(image, degrees):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


CONDITIONS = {
    "original": lambda im: im,
    "far2x": lambda im: shrink(im, 2),
    "far4x": lambda im: shrink(im, 4),
    "dark": lambda im: exposure(im, 0.45, 1.3),
    "verydark": lambda im: add_noise(exposure(im, 0.2, 1.5), 4),
    "bright": lambda im: exposure(im, 1.9, 0.7),
    "blur": lambda im: cv2.GaussianBlur(im, (0, 0), max(1.5, im.shape[1] / 600)),
    "noise": lambda im: add_noise(im, 18),
    "tilt15": lambda im: rotate(im, 15),
}


def read_image(path):
    image = cv2.imread(str(path))
    if image is not None:
        return image
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def main(folder):
    INPUT_DIR.mkdir(exist_ok=True)
    photos = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"})
    for path in photos:
        image = read_image(path)
        # Keep files a sensible size (large phone photos are slow to process).
        scale = 1600 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for condition, make in CONDITIONS.items():
            out = INPUT_DIR / f"{path.stem}_{condition}.jpg"
            cv2.imwrite(str(out), make(image), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{path.name}: {len(CONDITIONS)} versions saved")
    print(f"\nDone: {len(photos) * len(CONDITIONS)} photos in {INPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: make_conditions.py photos_folder")
    main(sys.argv[1])
