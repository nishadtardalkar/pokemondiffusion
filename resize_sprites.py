"""
Resize all sprites to a fixed resolution using nearest-neighbor sampling.

Reads PNGs from a sprites folder (e.g. 0001.png, 0002.png), resizes each to
the target size (default 128x128), and saves to an output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def resize_sprites(
    sprites_dir: Path,
    out_dir: Path,
    size: int,
) -> None:
    """Resize all sprites to size x size using nearest-neighbor sampling."""
    sprites_dir = sprites_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(sprites_dir.glob("*.png"))
    if not pngs:
        print(f"[WARN] No PNG files in {sprites_dir}")
        return

    for png_path in pngs:
        try:
            img = Image.open(png_path)
        except Exception as e:
            print(f"[ERROR] {png_path.name}: {e}")
            continue

        resized = img.resize((size, size), Image.Resampling.NEAREST)
        out_path = out_dir / png_path.name
        resized.save(out_path, "PNG")
        print(f"[INFO] {png_path.name} -> {size}x{size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all sprites to a fixed resolution using nearest-neighbor sampling.",
    )
    parser.add_argument(
        "--sprites-dir",
        type=Path,
        default=Path("sprites"),
        help="Directory containing pokemon sprites. Default: ./sprites",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sprites_128"),
        help="Output directory. Default: ./sprites_128",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Target resolution (width x height). Default: 128",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sprites_dir.is_dir():
        raise SystemExit(f"Sprites directory not found: {args.sprites_dir}")
    resize_sprites(args.sprites_dir, args.out_dir, args.size)


if __name__ == "__main__":
    main()
