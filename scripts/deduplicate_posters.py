"""Deduplicate poster images by perceptual hash.

Non-destructive: produces a filtered file list (and optionally a new
precomputed .npz) without moving or deleting any images.

For each group of visually identical posters, keeps the one with the
highest Letterboxd num_ratings.

Usage:
    python deduplicate_posters.py --images images_new --metadata poster_metadata.json
    python deduplicate_posters.py --images images_new --metadata poster_metadata.json --threshold 6 --write-npz grid_data_deduped.npz
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

# Module-level variable for worker processes
_worker_images_dir = None


def slug_from_fname(fname: str) -> str:
    """Extract slug from filename like '123_some-slug.jpg'."""
    base = os.path.splitext(fname)[0]
    parts = base.split("_", 1)
    return parts[1] if len(parts) == 2 else base


def _init_worker(images_dir):
    global _worker_images_dir
    _worker_images_dir = images_dir


def compute_phash(fname):
    """Compute perceptual hash for a single image. Returns (fname, hash_int) or None."""
    try:
        path = os.path.join(_worker_images_dir, fname)
        with Image.open(path) as img:
            h = imagehash.phash(img)
        # Return hash as int for pickling across processes
        return fname, int(str(h), 16)
    except Exception:
        return None


# ── Multi-probe LSH for fast hamming-distance neighbor search ───────

NUM_BANDS = 4
BAND_BITS = 16
BAND_MASK = (1 << BAND_BITS) - 1


def _extract_bands(h: int) -> list[int]:
    """Split a 64-bit hash into NUM_BANDS bands of BAND_BITS bits each."""
    return [(h >> (i * BAND_BITS)) & BAND_MASK for i in range(NUM_BANDS)]


def _band_probes(band_val: int) -> list[int]:
    """Return the band value itself plus all single-bit-flip neighbors."""
    probes = [band_val]
    for bit in range(BAND_BITS):
        probes.append(band_val ^ (1 << bit))
    return probes


def find_near_neighbors(hash_ints: list[int], threshold: int) -> list[tuple[int, int]]:
    """Find all pairs of hashes within hamming distance <= threshold.

    Uses multi-probe LSH: splits 64-bit hashes into 4 bands of 16 bits.
    By pigeonhole, if hamming(a,b) <= 6, at least one band differs by <= 1 bit.
    For each hash, probe exact + 16 single-bit-flip neighbors per band (68 lookups).
    Then verify candidates with exact hamming distance.
    """
    # Build band indexes: band_index -> {band_value -> set of hash indices}
    band_indexes = [defaultdict(set) for _ in range(NUM_BANDS)]
    all_bands = []
    for idx, h in enumerate(hash_ints):
        bands = _extract_bands(h)
        all_bands.append(bands)
        for b, val in enumerate(bands):
            band_indexes[b][val].add(idx)

    # Find candidate pairs via multi-probe
    pairs = []
    seen_pairs = set()
    for idx, h in enumerate(tqdm(hash_ints, desc="Near-dedup search")):
        candidates = set()
        for b, val in enumerate(all_bands[idx]):
            for probe in _band_probes(val):
                candidates.update(band_indexes[b].get(probe, set()))
        candidates.discard(idx)

        for cand_idx in candidates:
            pair = (min(idx, cand_idx), max(idx, cand_idx))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                dist = (h ^ hash_ints[cand_idx]).bit_count()
                if dist <= threshold:
                    pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Deduplicate poster images by perceptual hash.")
    parser.add_argument("--images", required=True, help="Directory containing poster JPEGs")
    parser.add_argument("--metadata", required=True, help="Path to poster_metadata.json")
    parser.add_argument(
        "--threshold",
        type=int,
        default=6,
        help="Hamming distance threshold for near-duplicates (default: 6)",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of threads (default: 16)")
    parser.add_argument("--output", default="deduplicated_files.json", help="Output JSON with kept/removed lists")
    parser.add_argument(
        "--write-npz",
        default=None,
        help="If set, write a precomputed grid_data .npz with only kept files",
    )
    parser.add_argument(
        "--source-npz",
        default=None,
        help="Source .npz to filter (required if --write-npz is set)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (for testing)")
    parser.add_argument(
        "--inspect",
        default=None,
        help="Directory to save visual comparison grids for near-dedup groups (for review)",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=50,
        help="Max number of groups to render for inspection (default: 50)",
    )
    args = parser.parse_args()

    images_dir = args.images
    if not os.path.isdir(images_dir):
        print(f"Error: {images_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Load metadata for rating-based keeper selection
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_by_slug = {m["slug"]: m for m in metadata}

    def get_ratings(fname):
        slug = slug_from_fname(fname)
        m = meta_by_slug.get(slug)
        if m and isinstance(m.get("num_ratings"), (int, float)):
            return m["num_ratings"]
        return 0

    # Collect image files
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"Found {len(files)} images")

    # Phase 1: Compute perceptual hashes using multiprocessing
    print("Computing perceptual hashes...")
    hashes = {}  # fname -> hash_hex_str

    with Pool(processes=args.workers, initializer=_init_worker, initargs=(images_dir,)) as pool:
        for result in tqdm(pool.imap_unordered(compute_phash, files, chunksize=64), total=len(files), desc="Hashing"):
            if result is not None:
                fname, h_int = result
                hashes[fname] = format(h_int, "016x")

    print(f"Hashed {len(hashes)} images")

    # Phase 2: Group exact hash matches first (fast dict lookup)
    exact_groups = defaultdict(list)
    for fname, h_hex in hashes.items():
        exact_groups[h_hex].append(fname)

    exact_dupes = sum(len(g) - 1 for g in exact_groups.values() if len(g) > 1)
    exact_groups_with_dupes = sum(1 for g in exact_groups.values() if len(g) > 1)
    print(f"\nExact hash matches: {exact_dupes} duplicates across {exact_groups_with_dupes} groups")

    # Phase 3: Near-duplicate grouping using multi-probe LSH + Union-Find
    if args.threshold > 0:
        print(f"\nSearching for near-duplicates (hamming <= {args.threshold})...")

        unique_hex = list(exact_groups.keys())
        unique_ints = [int(h, 16) for h in unique_hex]

        raw_pairs = find_near_neighbors(unique_ints, args.threshold)

        # Filter out near-dedup pairs involving low-information posters
        # (mostly dark/bright images with text that hash similarly but aren't duplicates)
        LOW_INFO_VAR = 2000  # images with pixel variance below this are low-info
        low_info_cache = {}

        def is_low_info_poster(fname):
            if fname not in low_info_cache:
                try:
                    img = np.array(Image.open(os.path.join(images_dir, fname)))
                    low_info_cache[fname] = img.var() < LOW_INFO_VAR
                except Exception:
                    low_info_cache[fname] = False
            return low_info_cache[fname]

        def group_is_low_info(h_hex):
            return all(is_low_info_poster(f) for f in exact_groups[h_hex])

        pairs = []
        filtered_out = 0
        for a, b in raw_pairs:
            if group_is_low_info(unique_hex[a]) or group_is_low_info(unique_hex[b]):
                filtered_out += 1
            else:
                pairs.append((a, b))
        if filtered_out:
            print(f"Filtered out {filtered_out} near-dedup pairs (low-info posters, var < {LOW_INFO_VAR})")

        # Union-Find to merge near-duplicate groups
        parent = list(range(len(unique_hex)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in pairs:
            union(a, b)

        # Build final groups from Union-Find
        final_groups = defaultdict(list)
        for idx, h_hex in enumerate(unique_hex):
            root = find(idx)
            final_groups[root].extend(exact_groups[h_hex])

        near_merges = len(pairs)
        print(f"Near-duplicate pairs found: {near_merges}")
    else:
        # No near-dedup, just use exact groups
        final_groups = exact_groups

    # Phase 4: Select keeper per group (highest num_ratings)
    kept = []
    removed = []
    group_sizes = []

    for group_files in final_groups.values():
        group_sizes.append(len(group_files))
        if len(group_files) == 1:
            kept.append(group_files[0])
        else:
            # Sort by num_ratings descending, keep the best
            ranked = sorted(group_files, key=get_ratings, reverse=True)
            kept.append(ranked[0])
            removed.extend(ranked[1:])

    kept.sort()
    removed.sort()

    # Stats
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total images:         {len(hashes):>8,}")
    print(f"Unique posters:       {len(kept):>8,}")
    print(f"Duplicates removed:   {len(removed):>8,}")
    print(f"Reduction:            {len(removed)/len(hashes)*100:>7.1f}%")
    print()

    # Group size distribution
    size_counts = defaultdict(int)
    for s in group_sizes:
        size_counts[s] += 1
    print("Group size distribution:")
    for size in sorted(size_counts.keys()):
        if size > 1 or size_counts[size] < 10:
            print(f"  {size:>3} copies: {size_counts[size]:>6,} groups")
        if size == 1:
            print(f"  {size:>3} copy:   {size_counts[size]:>6,} unique images")

    # Show some examples of large groups
    large_groups = sorted(
        [(g, max(g, key=get_ratings)) for g in final_groups.values() if len(g) > 3],
        key=lambda x: len(x[0]),
        reverse=True,
    )[:10]
    if large_groups:
        print(f"\nLargest duplicate groups:")
        for group, best in large_groups:
            best_slug = slug_from_fname(best)
            best_ratings = get_ratings(best)
            m = meta_by_slug.get(best_slug, {})
            name = m.get("name", best_slug)
            print(f"  {len(group):>3} copies: {name} ({best_ratings:,} ratings) - keeping {best}")

    # Visual inspection of near-dedup groups
    if args.inspect:
        os.makedirs(args.inspect, exist_ok=True)
        multi_groups = [
            (g, max(g, key=get_ratings))
            for g in final_groups.values()
            if len(g) > 1
        ]
        multi_groups.sort(key=lambda x: len(x[0]), reverse=True)
        multi_groups = multi_groups[: args.inspect_limit]

        print(f"\nRendering {len(multi_groups)} comparison grids to {args.inspect}/...")
        for gi, (group, best) in enumerate(tqdm(multi_groups, desc="Rendering")):
            imgs = []
            labels = []
            for fname in sorted(group, key=get_ratings, reverse=True):
                try:
                    img = Image.open(os.path.join(images_dir, fname))
                    imgs.append(img)
                    slug = slug_from_fname(fname)
                    r = get_ratings(fname)
                    label = f"{slug}\n{r:,} ratings"
                    if fname == best:
                        label += " [KEEP]"
                    labels.append(label)
                except Exception:
                    continue

            if len(imgs) < 2:
                continue

            # Build comparison grid: posters side-by-side with labels
            thumb_w, thumb_h = 230, 345
            label_h = 40
            cols = min(len(imgs), 8)
            rows = (len(imgs) + cols - 1) // cols
            grid_w = cols * thumb_w
            grid_h = rows * (thumb_h + label_h)
            grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

            from PIL import ImageDraw

            draw = ImageDraw.Draw(grid)
            for i, (img, label) in enumerate(zip(imgs, labels)):
                col = i % cols
                row = i // cols
                x = col * thumb_w
                y = row * (thumb_h + label_h)
                resized = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                grid.paste(resized, (x, y))
                # Draw label below
                text_y = y + thumb_h + 2
                color = (0, 255, 0) if "[KEEP]" in label else (200, 200, 200)
                draw.text((x + 4, text_y), label.split("\n")[0][:30], fill=color)
                draw.text((x + 4, text_y + 14), label.split("\n")[1] if "\n" in label else "", fill=color)

            grid.save(os.path.join(args.inspect, f"group_{gi:03d}_{len(group)}copies.jpg"), quality=85)

        print(f"Saved {len(multi_groups)} comparison grids")

    # Save results
    output_data = {
        "kept": kept,
        "removed": removed,
        "stats": {
            "total": len(hashes),
            "unique": len(kept),
            "duplicates": len(removed),
            "threshold": args.threshold,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved dedup results to {args.output}")

    # Optionally filter an existing .npz
    if args.write_npz:
        if not args.source_npz:
            print("Error: --source-npz required when using --write-npz", file=sys.stderr)
            sys.exit(1)
        print(f"\nFiltering {args.source_npz} -> {args.write_npz}...")
        data = np.load(args.source_npz, allow_pickle=True)
        src_filenames = list(data["filenames"])
        src_vectors = data["vectors"]

        kept_set = set(kept)
        mask = [f in kept_set for f in src_filenames]
        new_filenames = np.array([f for f, m in zip(src_filenames, mask) if m])
        new_vectors = src_vectors[mask]

        np.savez_compressed(args.write_npz, vectors=new_vectors, filenames=new_filenames)
        print(f"Saved {len(new_filenames)} vectors to {args.write_npz} (was {len(src_filenames)})")


if __name__ == "__main__":
    main()
