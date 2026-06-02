#!/usr/bin/env python
"""Validate HDF5-backed BioModelML image shards against their manifest."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py

from biomodelml.data_generation.hdf5_store import validate_hdf5_image_file


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    """Resolve a manifest path that may be absolute or relative."""
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else root_dir / candidate


def validate_dataset(root_dir: Path) -> None:
    """Validate the full generated dataset rooted at output_dir."""
    manifest_path = root_dir / "metadata" / "image_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)

    images = manifest.get("images", [])
    if not images:
        raise ValueError("Manifest contains no images to validate")

    by_file: Dict[Path, List[dict]] = {}
    for entry in images:
        image_path = _resolve_path(root_dir, entry["image_path"])
        by_file.setdefault(image_path, []).append(entry)

    total_validated = 0
    for shard_path, shard_entries in by_file.items():
        report = validate_hdf5_image_file(shard_path, expected_count=len(shard_entries))

        with h5py.File(shard_path, "r") as handle:
            for entry in shard_entries:
                dataset_path = entry.get("dataset_path")
                if not dataset_path:
                    raise ValueError(f"Missing dataset_path in manifest entry {entry['image_id']}")
                if dataset_path not in handle:
                    raise ValueError(
                        f"Dataset {dataset_path} referenced by {entry['image_id']} not found in {shard_path}"
                    )

                dataset = handle[dataset_path]
                if dataset.attrs.get("image_id") != entry["image_id"]:
                    raise ValueError(
                        f"Image ID mismatch for {entry['image_id']} in {shard_path}:{dataset_path}"
                    )

                if dataset.attrs.get("matrix_checksum") != entry.get("matrix_checksum"):
                    raise ValueError(
                        f"Checksum mismatch for {entry['image_id']} in {shard_path}:{dataset_path}"
                    )

        total_validated += report.validated_count
        print(f"✓ Validated {report.validated_count} datasets in {shard_path}")

    expected_total = len(images)
    if total_validated != expected_total:
        raise ValueError(f"Validated {total_validated} datasets but manifest contains {expected_total}")

    print(f"✓ Validation complete: {total_validated} datasets across {len(by_file)} HDF5 shards")


def main() -> None:
    """Validate generated HDF5 image shards."""
    parser = argparse.ArgumentParser(
        description="Validate BioModelML HDF5 image output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  biomodelml-validate-images output/
        """,
    )

    parser.add_argument("root_dir", help="Root output directory created by biomodelml-generate-images")

    args = parser.parse_args()

    try:
        validate_dataset(Path(args.root_dir))
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()