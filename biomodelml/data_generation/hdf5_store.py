"""HDF5 helpers for storing and validating generated BioModelML matrices."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from biomodelml.structs import ImageMetadata


def matrix_checksum(matrix: np.ndarray) -> str:
    """Return a stable checksum for a matrix payload."""
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def safe_dataset_name(image_id: str) -> str:
    """Normalize an image id into a valid HDF5 dataset name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_id).strip("_")
    return cleaned or "image"


@dataclass
class HDF5ValidationReport:
    """Summary of an HDF5 validation pass."""

    file_path: str
    dataset_count: int
    expected_count: Optional[int]
    validated_count: int


class HDF5ImageWriter:
    """Write generated matrices to a sharded HDF5 file."""

    def __init__(
        self,
        file_path: Path,
        source_fasta: str,
        sequence_type: str,
        max_window: int,
        num_workers: int,
        include_metadata: bool,
    ):
        self.file_path = Path(file_path)
        self.sequence_type = sequence_type
        self._handle = h5py.File(self.file_path, "w")
        self._handle.attrs["created_at"] = datetime.now().isoformat()
        self._handle.attrs["source_fasta"] = source_fasta
        self._handle.attrs["sequence_type"] = sequence_type
        self._handle.attrs["max_window"] = max_window
        self._handle.attrs["num_workers"] = num_workers
        self._handle.attrs["include_metadata"] = include_metadata
        self._handle.attrs["storage_format"] = "hdf5"
        self._handle.attrs["dataset_count"] = 0

    def __enter__(self) -> "HDF5ImageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(validate=exc_type is None)
        return False

    def write_matrix(
        self,
        image_id: str,
        matrix: np.ndarray,
        source_fasta: str,
        sequence_name: str,
        sequence_length: int,
        tree_distances_path: Optional[str] = None,
    ) -> ImageMetadata:
        """Store one matrix dataset and return its metadata record."""
        dataset_name = self._unique_dataset_name(safe_dataset_name(image_id))
        checksum = matrix_checksum(matrix)

        dataset = self._handle.create_dataset(
            dataset_name,
            data=matrix,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=True,
        )
        dataset.attrs["image_id"] = image_id
        dataset.attrs["source_fasta"] = source_fasta
        dataset.attrs["sequence_name"] = sequence_name
        dataset.attrs["sequence_length"] = sequence_length
        dataset.attrs["matrix_checksum"] = checksum
        dataset.attrs["tree_distances_path"] = tree_distances_path or ""
        dataset.attrs["tree_distances_available"] = bool(tree_distances_path)
        dataset.attrs["storage_format"] = "hdf5"

        self._handle.attrs["dataset_count"] = int(self._handle.attrs["dataset_count"]) + 1

        return ImageMetadata(
            image_id=image_id,
            image_path=str(self.file_path),
            source_fasta=source_fasta,
            sequence_name=sequence_name,
            sequence_length=sequence_length,
            matrix_shape=tuple(matrix.shape),
            tree_distances_path=tree_distances_path,
            tree_distances_available=tree_distances_path is not None,
            dataset_path=dataset_name,
            matrix_checksum=checksum,
            storage_format="hdf5",
        )

    def close(self, validate: bool = True) -> None:
        """Flush and optionally validate the shard before closing."""
        if self._handle is None:
            return

        self._handle.flush()
        self._handle.close()
        self._handle = None

        if validate:
            validate_hdf5_image_file(self.file_path)

    def _unique_dataset_name(self, base_name: str) -> str:
        """Avoid dataset collisions within a shard."""
        candidate = base_name
        suffix = 2
        while candidate in self._handle:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate


def validate_hdf5_image_file(
    file_path: Path,
    expected_count: Optional[int] = None,
) -> HDF5ValidationReport:
    """Validate that an HDF5 shard contains well-formed RGB matrices."""
    validated_count = 0
    with h5py.File(file_path, "r") as handle:
        dataset_names = [name for name, item in handle.items() if isinstance(item, h5py.Dataset)]
        if expected_count is None:
            expected_count = int(handle.attrs.get("dataset_count", len(dataset_names)))

        if len(dataset_names) != expected_count:
            raise ValueError(
                f"Validation failed for {file_path}: expected {expected_count} datasets, "
                f"found {len(dataset_names)}"
            )

        for dataset_name in dataset_names:
            dataset = handle[dataset_name]
            matrix = dataset[...]

            if matrix.dtype != np.uint8:
                raise ValueError(
                    f"Validation failed for {file_path}:{dataset_name} - expected uint8, got {matrix.dtype}"
                )

            if matrix.ndim != 3 or matrix.shape[2] != 3:
                raise ValueError(
                    f"Validation failed for {file_path}:{dataset_name} - expected shape (H, W, 3), got {matrix.shape}"
                )

            stored_checksum = dataset.attrs.get("matrix_checksum")
            actual_checksum = matrix_checksum(matrix)
            if stored_checksum != actual_checksum:
                raise ValueError(
                    f"Validation failed for {file_path}:{dataset_name} - checksum mismatch"
                )

            validated_count += 1

    return HDF5ValidationReport(
        file_path=str(file_path),
        dataset_count=validated_count,
        expected_count=expected_count,
        validated_count=validated_count,
    )