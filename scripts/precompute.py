"""Precompute sRGB grid color vectors for all tile images.

Each image is divided into a 10x15 grid of equal cells. The average
RGB color of each cell is stored as a 450-dimensional feature vector
(150 cells x 3 channels) per image.

Usage:
    python scripts/precompute.py --images path/to/tiles
    python scripts/precompute.py --images path/to/tiles --tile-size 300x300
    python scripts/precompute.py --images path/to/tiles --metadata data/poster_metadata.json --min-ratings 3500
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

GRID_COLS = 10
GRID_ROWS = 15

# Module-level tile dimensions (set from CLI args before workers start)
_TILE_W = 230
_TILE_H = 345


def process_one(args_tuple):
    """Load and compute vector for a single image. Returns (fname, vector) or None."""
    images_dir, fname = args_tuple
    path = os.path.join(images_dir, fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != _TILE_W or h != _TILE_H:
        return None
    cell_w = _TILE_W // GRID_COLS
    cell_h = _TILE_H // GRID_ROWS
    # cv2 loads BGR; convert to RGB then compute vector
    pixels = img[:, :, ::-1].astype(np.float32)
    grid = pixels.reshape(GRID_ROWS, cell_h, GRID_COLS, cell_w, 3)
    cell_avgs = grid.mean(axis=(1, 3))
    return fname, cell_avgs.reshape(-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Precompute sRGB grid vectors for poster images.")
    parser.add_argument(
        "--images",
        default=os.environ.get("MOSAIC_IMAGES_DIR", "images"),
        help="Directory containing tile images (default: $MOSAIC_IMAGES_DIR or 'images/')",
    )
    parser.add_argument(
        "--output",
        default="data/grid_data.npz",
        help="Output file path (default: data/grid_data.npz)",
    )
    parser.add_argument(
        "--tile-size",
        default="230x345",
        help="Tile dimensions as WxH (default: 230x345). Width must be divisible by 10, height by 15.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to poster_metadata.json for filtering",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=0,
        help="Minimum Letterboxd num_ratings to include a poster (requires --metadata)",
    )
    parser.add_argument(
        "--genre",
        default=None,
        help="Only include posters matching this genre (case-insensitive, requires --metadata)",
    )
    args = parser.parse_args()

    # Parse and validate tile size
    global _TILE_W, _TILE_H
    try:
        _TILE_W, _TILE_H = (int(x) for x in args.tile_size.lower().split("x"))
    except ValueError:
        print(f"Error: invalid --tile-size '{args.tile_size}', expected WxH (e.g. 230x345)", file=sys.stderr)
        sys.exit(1)
    if _TILE_W % GRID_COLS != 0 or _TILE_H % GRID_ROWS != 0:
        print(f"Error: tile width must be divisible by {GRID_COLS} and height by {GRID_ROWS}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.images):
        print(f"Error: image directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    # Collect all image files
    files = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(files)} images in {args.images}")

    # Apply metadata filters
    if args.metadata and (args.min_ratings or args.genre):
        with open(args.metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Build lookup: slug -> metadata entry
        meta_by_slug = {m["slug"]: m for m in metadata}

        # Extract slug from filename: "{index}_{slug}.jpg"
        def slug_from_fname(fname):
            base = os.path.splitext(fname)[0]
            parts = base.split("_", 1)
            return parts[1] if len(parts) == 2 else base

        before = len(files)
        filtered = []
        for f in files:
            slug = slug_from_fname(f)
            m = meta_by_slug.get(slug)
            if m is None:
                continue
            if args.min_ratings:
                nr = m.get("num_ratings", 0)
                if not isinstance(nr, (int, float)) or nr < args.min_ratings:
                    continue
            if args.genre:
                genres = m.get("genre", [])
                if isinstance(genres, list):
                    if not any(args.genre.lower() in g.lower() for g in genres):
                        continue
                elif isinstance(genres, str):
                    if args.genre.lower() not in genres.lower():
                        continue
                else:
                    continue
            filtered.append(f)
        files = filtered
        print(f"Metadata filter: {before} -> {len(files)} images (min_ratings={args.min_ratings}, genre={args.genre})")

    num_workers = min(os.cpu_count() or 4, 16)
    work_items = [(args.images, f) for f in files]

    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_one, work_items),
            total=len(files),
            desc="Processing posters",
        ):
            if result is not None:
                fname, vec = result
                results[fname] = vec

    if not results:
        print(f"Error: no valid {_TILE_W}x{_TILE_H} images found.", file=sys.stderr)
        sys.exit(1)

    # Sort by filename to maintain deterministic order
    filenames = sorted(results.keys())
    vectors = [results[f] for f in filenames]

    vectors_arr = np.stack(vectors)  # (N, 450) float32
    filenames_arr = np.array(filenames)

    np.savez_compressed(
        args.output,
        vectors=vectors_arr,
        filenames=filenames_arr,
        tile_size=np.array([_TILE_W, _TILE_H]),
    )
    print(f"Saved {len(filenames)} vectors ({_TILE_W}x{_TILE_H} tiles) to {args.output}")
    skipped = len(files) - len(filenames)
    if skipped:
        print(f"Skipped {skipped} files (wrong size or unreadable)")


if __name__ == "__main__":
    main()
