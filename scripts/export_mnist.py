#!/usr/bin/env python3
"""Download real MNIST and export downsampled digits for the Lean example.

Produces ``data/mnist.csv`` — a simple CSV consumed by
``examples/VerificationMNIST.lean``.

Each line:  ``split,label,v0,v1,...,v{dim-1}``

* Images are average-pooled from 28×28 to a configurable grid (default 8×8 = 64).
* Pixel values are normalised to [−1, +1] (centred around 0).
* A configurable number of samples per class is exported
  (default: 5 train + 2 test per class).

Usage::

    python scripts/export_mnist.py                  # 8×8 (default)
    python scripts/export_mnist.py --rows 4 --cols 8   # 4×8 = 32
    python scripts/export_mnist.py --rows 14 --cols 14 # 14×14 = 196
    python scripts/export_mnist.py --train-per-class 10 --test-per-class 5
    python scripts/export_mnist.py --print-random 3    # show 3 random samples

Requirements: only the Python standard library (no torch/numpy).
MNIST is fetched from Yann LeCun's site via ``urllib``.
"""

from __future__ import annotations

import argparse
import gzip
import random
import struct
import urllib.request

from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download(filename: str) -> Path:
    """Download a gzipped MNIST file if not already cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / filename
    if dest.exists():
        return dest
    url = f"{MNIST_URL}/{filename}"
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, dest)
    return dest


def read_images(path: Path) -> list[list[int]]:
    """Read IDX image file → list of flat pixel lists (0-255)."""
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = f.read()
    images: list[list[int]] = []
    stride = rows * cols
    for i in range(n):
        images.append(list(pixels[i * stride : (i + 1) * stride]))
    return images


def read_labels(path: Path) -> list[int]:
    """Read IDX label file → list of int labels."""
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
        labels = list(f.read(n))
    return labels


# ---------------------------------------------------------------------------
# Downsampling (average pooling)
# ---------------------------------------------------------------------------

def downsample(  # pylint: disable=too-many-locals
    img: list[int],
    target_rows: int,
    target_cols: int,
    src_h: int = 28,
    src_w: int = 28,
) -> list[float]:
    """Average-pool a 28×28 image to *target_rows* × *target_cols*.

    Returns values normalised to [−1, +1].
    """
    bh = src_h // target_rows
    bw = src_w // target_cols
    # Use non-overlapping blocks; we may crop a few rightmost columns
    out: list[float] = []
    for r in range(target_rows):
        for c in range(target_cols):
            total = 0.0
            count = 0
            for dr in range(bh):
                for dc in range(bw):
                    y = r * bh + dr
                    x = c * bw + dc
                    if y < src_h and x < src_w:
                        total += img[y * src_w + x]
                        count += 1
            avg = total / max(count, 1)
            # Normalise 0-255 → [−1, +1]
            out.append(avg / 127.5 - 1.0)
    return out


# ---------------------------------------------------------------------------
# ASCII visualisation
# ---------------------------------------------------------------------------

_SHADES = " ░▒▓█"


def display_ascii(
    pixels: list[float], rows: int, cols: int, label: int
) -> None:
    """Render a downsampled image as ASCII art in the terminal."""
    print(f"\n  Random sample — label: {label}  ({rows}×{cols} = {rows * cols} dims)")
    print(f"  ┌{'─' * (cols * 2)}┐")
    for r in range(rows):
        line = ""
        for c in range(cols):
            v = pixels[r * cols + c]          # in [−1, +1]
            idx = int((v + 1.0) / 2.0 * (len(_SHADES) - 1) + 0.5)
            idx = max(0, min(len(_SHADES) - 1, idx))
            line += _SHADES[idx] * 2
        print(f"  │{line}│")
    print(f"  └{'─' * (cols * 2)}┘")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Download MNIST and export downsampled digits as CSV."
    )
    p.add_argument(
        "--rows", type=int, default=8,
        help="Number of rows in the downsampled image (default: 8)",
    )
    p.add_argument(
        "--cols", type=int, default=8,
        help="Number of columns in the downsampled image (default: 8)",
    )
    p.add_argument(
        "--train-per-class", type=int, default=5,
        help="Number of training samples per class (default: 5)",
    )
    p.add_argument(
        "--test-per-class", type=int, default=2,
        help="Number of test samples per class (default: 2)",
    )
    p.add_argument(
        "--print-random", type=int, default=1,
        help="Number of random samples to display (default: 1, 0 to disable)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # pylint: disable=too-many-locals
    """Main entry point."""
    args = parse_args()
    target_rows = args.rows
    target_cols = args.cols
    train_per_class = args.train_per_class
    test_per_class = args.test_per_class
    ndims = target_rows * target_cols

    # Download
    train_img_path = download(FILES["train_images"])
    train_lbl_path = download(FILES["train_labels"])
    test_img_path = download(FILES["test_images"])
    test_lbl_path = download(FILES["test_labels"])

    train_images = read_images(train_img_path)
    train_labels = read_labels(train_lbl_path)
    test_images = read_images(test_img_path)
    test_labels = read_labels(test_lbl_path)

    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)

    # Collect per-class samples
    train_by_class: dict[int, list[list[int]]] = {c: [] for c in range(10)}
    test_by_class: dict[int, list[list[int]]] = {c: [] for c in range(10)}

    for img, lbl in zip(train_images, train_labels):
        if len(train_by_class[lbl]) < train_per_class:
            train_by_class[lbl].append(img)
    for img, lbl in zip(test_images, test_labels):
        if len(test_by_class[lbl]) < test_per_class:
            test_by_class[lbl].append(img)

    # Write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mnist.csv"
    written = 0
    all_rows: list[tuple[int, list[float]]] = []  # for random display
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Real MNIST downsampled to {ndims} dims "
            f"({target_rows}x{target_cols} average pool)\n"
        )
        f.write(f"# Format: split,label,v0,v1,...,v{ndims - 1}\n")
        for cls in range(10):
            for img in train_by_class[cls]:
                row = downsample(img, target_rows, target_cols)
                vals = ",".join(f"{v:.6f}" for v in row)
                f.write(f"train,{cls},{vals}\n")
                all_rows.append((cls, row))
                written += 1
        for cls in range(10):
            for img in test_by_class[cls]:
                row = downsample(img, target_rows, target_cols)
                vals = ",".join(f"{v:.6f}" for v in row)
                f.write(f"test,{cls},{vals}\n")
                all_rows.append((cls, row))
                written += 1

    print(
        f"Wrote {written} samples ({target_rows}×{target_cols} = "
        f"{ndims} dims) to {out_path}"
    )

    # Display random samples
    n_print = min(args.print_random, len(all_rows))
    if n_print > 0 and all_rows:
        samples = random.sample(all_rows, n_print)
        for lbl, pixels in samples:
            display_ascii(pixels, target_rows, target_cols, lbl)


if __name__ == "__main__":
    main()
