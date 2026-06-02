#!/usr/bin/env python
"""CLI command for visualizing a saved BioModelML matrix image."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import h5py
import numpy as np


def _load_matrix(image_path: Path, dataset_path: Optional[str] = None) -> np.ndarray:
    """Load a matrix stored as .h5/.hdf5, or legacy .npy/.npz files."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(image_path, "r") as handle:
            target_dataset = dataset_path
            if target_dataset is None:
                dataset_names = list(handle.keys())
                if len(dataset_names) != 1:
                    raise ValueError(
                        f"HDF5 file {image_path} contains multiple datasets; "
                        "use --dataset to select one"
                    )
                target_dataset = dataset_names[0]

            if target_dataset not in handle:
                raise ValueError(f"Dataset '{target_dataset}' not found in {image_path}")

            return handle[target_dataset][...]

    loaded = np.load(image_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "matrix" not in loaded.files:
            raise ValueError(f"{image_path} does not contain a 'matrix' array")
        return loaded["matrix"]
    return loaded


def _select_view(matrix: np.ndarray, view: str) -> np.ndarray:
    """Select the data to display based on the requested view."""
    if matrix.ndim != 3 or matrix.shape[2] != 3:
        raise ValueError(f"Expected an RGB matrix with shape (H, W, 3), got {matrix.shape}")

    if view == "rgb":
        return matrix

    if view == "gray":
        return matrix.mean(axis=2)

    channel_map = {"red": 0, "green": 1, "blue": 2}
    return matrix[:, :, channel_map[view]]


def main() -> None:
    """Visualize a matrix saved by BioModelML."""
    parser = argparse.ArgumentParser(
                description="Visualize a BioModelML matrix image from HDF5 or legacy NumPy storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    biomodelml-visualize-image output/images/alignment_001.h5
    biomodelml-visualize-image output/images/alignment_001.h5 --dataset alignment_001_seq1 --view red
    biomodelml-visualize-image output/images/alignment_001.h5 --save alignment_001.png
        """,
    )

        parser.add_argument("image_file", help="Path to a .h5/.hdf5, .npy, or .npz matrix file")
    parser.add_argument(
        "--view",
        choices=["rgb", "red", "green", "blue", "gray"],
        default="rgb",
        help="How to display the matrix (default: rgb)",
    )
    parser.add_argument(
        "--save",
        help="Optional path to save the visualization as a PNG file",
    )
    parser.add_argument(
        "--dataset",
        help="HDF5 dataset name when visualizing a .h5/.hdf5 file with multiple datasets",
    )
    parser.add_argument(
        "--title",
        help="Optional plot title",
    )

    args = parser.parse_args()

    image_path = Path(args.image_file)

    try:
        matrix = _load_matrix(image_path, args.dataset)
        view_data = _select_view(matrix, args.view)
    except Exception as e:
        print(f"Error loading image matrix: {e}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 8))

    if args.view == "rgb":
        ax.imshow(view_data)
    elif args.view == "gray":
        ax.imshow(view_data, cmap="gray")
    elif args.view == "red":
        ax.imshow(view_data, cmap="Reds")
    elif args.view == "green":
        ax.imshow(view_data, cmap="Greens")
    else:
        ax.imshow(view_data, cmap="Blues")

    ax.set_axis_off()
    ax.set_title(args.title or image_path.name)
    fig.tight_layout()

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()