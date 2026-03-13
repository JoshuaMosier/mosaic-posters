# Mosaic Posters

Generate photomosaics from a library of movie poster images. Given a reference image, the tool finds the best-matching poster for each cell in a grid, producing a high-resolution mosaic.

<p align="center">
  <img src="docs/reference.jpg" width="200" alt="Reference image">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/overview.jpg" width="200" alt="100x100 mosaic">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/zoom.jpg" width="200" alt="Zoomed in to show individual poster tiles">
</p>
<p align="center">
  <em>Reference &rarr; 100x100 mosaic (10,000 unique posters) &rarr; zoomed detail</em>
</p>

## How it works

The pipeline has two steps:

### 1. Precompute poster vectors

Each 230x345 poster is divided into a 10x15 grid of 23x23px cells. The average sRGB color of each cell is stored as a 450-dimensional feature vector (150 cells x 3 channels). This only needs to run once for a given set of posters.

```
python precompute.py --images path/to/posters --output grid_data.npz
```

### 2. Build a mosaic

The reference image is center-cropped to 2:3 aspect ratio and divided into a grid. Each cell is resized to 230x345 with Lanczos interpolation, converted to the same 450D vector, and matched to the nearest unused poster by Euclidean distance.

```
python mosaic.py --reference photo.jpg --data grid_data.npz --images path/to/posters --cells 80 --output mosaic.jpg
```

| Flag | Default | Description |
|------|---------|-------------|
| `--reference` | *(required)* | Path to the reference image |
| `--data` | `grid_data.npz` | Precomputed vectors from step 1 |
| `--images` | `$MOSAIC_IMAGES_DIR` or `images/` | Directory of 230x345 poster JPEGs |
| `--cells` | `30` | Number of columns in the mosaic grid |
| `--rows` | auto (1.5x cells) | Number of rows (override for non-2:3 tiles) |
| `--output` | `mosaic.jpg` | Output file path |

## Poster images

The poster library should be a flat directory of 230x345 JPEG files. Images that don't match this exact size are skipped during precomputation. With ~31k posters, grids up to ~170x170 can use a unique poster per cell.

## Performance

Benchmarked with 31,764 posters on a 16-core machine:

| Step | Time |
|------|------|
| Precompute vectors | ~17s |
| Build 80x80 mosaic (6,400 tiles) | ~18s |
| Build 100x100 mosaic (10,000 tiles) | ~28s |

Key optimizations:
- **Threaded I/O** for JPEG loading via OpenCV (releases the GIL)
- **BLAS matrix multiplication** for distance computation instead of KD-tree (faster in 450D)
- **Partial sort** (`argpartition`) to rank only the top-1000 candidates per cell
- **BGR passthrough** in assembly to avoid color-swapping the full mosaic array

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.10+.
