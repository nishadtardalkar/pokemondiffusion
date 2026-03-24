"""
Build a multi-resolution dataset from Pokémon sprites.

Reads PNGs from a sprites folder (e.g. 0001.png, 0002.png), then for each image
creates downscaled versions by repeatedly halving the size down to 1px, saving:

  dataset/<resolution>/<pokemonid>.png

Downsampling: palette-based in HSV. We find a small set of dominant colors
for the whole sprite, then for each output pixel choose the nearest palette
color from its source region. Transparent pixels are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,255] to HSV H,S,V in [0,1]. Shape (..., 3) -> (..., 3)."""
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    s = np.where(mx > 0, (mx - mn) / mx, 0.0)
    diff = mx - mn
    rc = np.where(mx != mn, (mx - r) / np.where(diff > 0, diff, 1), 0)
    gc = np.where(mx != mn, (mx - g) / np.where(diff > 0, diff, 1), 0)
    bc = np.where(mx != mn, (mx - b) / np.where(diff > 0, diff, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV H,S,V in [0,1] to RGB [0,255]. Shape (..., 3) -> (..., 3)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = (h * 6.0) % 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255).clip(0, 255).astype(np.uint8)


def downsample_palette_hsv(
    arr: np.ndarray,
    target_size: int,
    k: int = 8,
    max_iter: int = 10,
) -> np.ndarray:
    """
    Palette-based downsampling in HSV.

    1. Convert the (cropped) sprite to HSV and collect all opaque pixels.
    2. Run a small k-means in HSV to find k dominant colors (the palette).
    3. For each output pixel, look at the corresponding source block,
       find which palette color best represents that block, and paint it.
    """
    h, w = arr.shape[:2]
    nch = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    base = min(h, w)
    arr = arr[:base, :base]
    if base < 1 or target_size < 1:
        return arr

    # Build HSV + opacity mask
    hsv = rgb_to_hsv(arr[..., :3])
    opaque = np.ones((base, base), dtype=bool)
    if nch >= 4:
        opaque = arr[..., 3] >= 128

    data = hsv[opaque]
    if data.size == 0:
        # Entire sprite is transparent
        return np.zeros((target_size, target_size, nch), dtype=arr.dtype)

    # k-means on HSV of opaque pixels
    n_points = data.shape[0]
    k_eff = min(k, n_points)
    # Randomly pick initial centroids
    rng = np.random.default_rng()
    init_idx = rng.choice(n_points, size=k_eff, replace=False)
    centroids = data[init_idx]

    for _ in range(max_iter):
        # Assign points to nearest centroid
        diff = data[:, None, :] - centroids[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        labels = np.argmin(dist2, axis=1)

        new_centroids = np.empty_like(centroids)
        for ci in range(k_eff):
            mask = labels == ci
            if not np.any(mask):
                # Keep old centroid if no points assigned
                new_centroids[ci] = centroids[ci]
            else:
                new_centroids[ci] = data[mask].mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids


    # Downsample: map each output cell to block in original and snap to palette
    block = base // target_size
    if block < 1:
        block = 1
    out_hsv = np.zeros((target_size, target_size, 3), dtype=hsv.dtype)
    out_alpha = np.zeros((target_size, target_size), dtype=arr.dtype)

    for oy in range(target_size):
        for ox in range(target_size):
            y0 = oy * block
            x0 = ox * block
            y1 = min(y0 + block, base)
            x1 = min(x0 + block, base)

            region_mask = opaque[y0:y1, x0:x1]
            if not np.any(region_mask):
                # Leave transparent
                continue

            region_hsv = hsv[y0:y1, x0:x1][region_mask]
            # Assign region pixels to nearest palette color
            diff = region_hsv[:, None, :] - centroids[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            region_labels = np.argmin(dist2, axis=1)

            # Pick the most frequent palette index in this region
            counts = np.bincount(region_labels, minlength=k_eff)
            palette_idx = int(np.argmax(counts))

            out_hsv[oy, ox] = centroids[palette_idx]
            out_alpha[oy, ox] = 255

    out_rgb = hsv_to_rgb(out_hsv)
    if nch >= 4:
        return np.concatenate(
            [out_rgb.astype(arr.dtype), out_alpha[..., None].astype(arr.dtype)],
            axis=-1,
        )
    return out_rgb.astype(arr.dtype)


def resolution_chain(base: int) -> list[int]:
    """Return [base, base//2, base//4, ...] down to 1."""
    if base < 1:
        return []
    out = [base]
    while base > 1:
        base = base // 2
        out.append(base)
    return out


def build_dataset(sprites_dir: Path, dataset_dir: Path) -> None:
    """Generate dataset/<res>/<id>.png for each sprite and each resolution."""
    sprites_dir = sprites_dir.resolve()
    dataset_dir = dataset_dir.resolve()

    pngs = sorted(sprites_dir.glob("*.png"))
    if not pngs:
        print(f"[WARN] No PNG files in {sprites_dir}")
        return

    for png_path in pngs:
        pokemon_id = png_path.stem  # e.g. 0001
        try:
            img = Image.open(png_path).convert("RGBA")
        except Exception as e:
            print(f"[ERROR] {png_path.name}: {e}")
            continue

        w, h = img.size
        base = min(w, h)
        if base < 1:
            continue

        sizes = resolution_chain(base)

        # Crop to center square (base x base) — use this for every target resolution
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        offset_h = (h - base) // 2
        offset_w = (w - base) // 2
        original = arr[offset_h : offset_h + base, offset_w : offset_w + base]

        for size in sizes:
            out_folder = dataset_dir / str(size)
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / f"{pokemon_id}.png"

            if size == base:
                out_arr = original
            else:
                out_arr = downsample_palette_hsv(original, size)

            # Keep channel dimension so 1x1 RGBA stays 1x1x4, not 1x4
            out_img = Image.fromarray(out_arr)
            out_img.save(out_path, "PNG")

        print(f"[INFO] {pokemon_id}: {sizes[0]} -> ... -> 1 ({len(sizes)} resolutions)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset/<resolution>/<pokemonid>.png from sprites by halving resolution down to 1px.",
    )
    parser.add_argument(
        "--sprites-dir",
        type=Path,
        default=Path("sprites_128"),
        help="Directory containing pokemon sprites (e.g. 0001.png). Default: ./sprites_128",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset root. Default: ./dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sprites_dir.is_dir():
        raise SystemExit(f"Sprites directory not found: {args.sprites_dir}")
    build_dataset(args.sprites_dir, args.dataset_dir)


if __name__ == "__main__":
    main()
