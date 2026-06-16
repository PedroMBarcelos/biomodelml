"""
Training dataset loader for deep learning workflows.

Provides PyTorch-compatible dataset classes for loading generated images
and paired ground-truth tree distances.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import h5py
import numpy as np
import pandas as pd


class ImageTreePair:
    """Single training sample: image array + ground-truth distances."""
    
    def __init__(
        self,
        image_id: str,
        image_array: np.ndarray,
        distances: Optional[np.ndarray] = None,
        sequence_name: Optional[str] = None,
        sequence_length: Optional[int] = None,
    ):
        """
        Initialize training sample.
        
        Args:
            image_id: Unique identifier for this sample
            image_array: RGB matrix (H, W, 3) as uint8
            distances: Pairwise distances vector (optional)
            sequence_name: Name of sequence (optional)
            sequence_length: Length of original sequence (optional)
        """
        self.image_id = image_id
        self.image_array = image_array
        self.distances = distances
        self.sequence_name = sequence_name
        self.sequence_length = sequence_length
    
    def __repr__(self) -> str:
        image_shape = None if self.image_array is None else self.image_array.shape
        return (
            f"ImageTreePair(id={self.image_id}, "
            f"image_shape={image_shape}, "
            f"has_distances={self.distances is not None})"
        )


class TrainingDataset:
    """Load and manage training dataset with images + ground-truth distances."""
    
    def __init__(self, root_dir: str, lazy_load: bool = True):
        """
        Initialize training dataset loader.
        
        Args:
            root_dir: Root directory containing metadata/images/ subdirectories
            lazy_load: Load images on-demand (default: True, memory efficient)
            
        Raises:
            FileNotFoundError: If manifest file not found
        """
        self.root_dir = Path(root_dir)
        self.lazy_load = lazy_load
        
        # Load manifest
        manifest_file = self.root_dir / "metadata" / "image_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Image manifest not found at {manifest_file}. "
                f"Run biomodelml-generate-images first."
            )
        
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        self.samples: List[ImageTreePair] = []
        self.distance_caches: Dict[str, np.ndarray] = {}
        
        # Build sample index
        self._build_sample_index()
    
    def _build_sample_index(self) -> None:
        """Build list of samples from manifest."""
        for img_data in self.manifest.get("images", []):
            image_id = img_data["image_id"]
            raw_image_path = Path(img_data["image_path"])
            image_path = raw_image_path if raw_image_path.is_absolute() else self.root_dir / raw_image_path
            dataset_path = img_data.get("dataset_path")
            
            # Load distances if available
            distances = None
            if img_data.get("tree_distances_available", False):
                distances_path = img_data.get("tree_distances_path")
                if distances_path:
                    distances = self._load_distances_for_image(distances_path, img_data["sequence_name"])
            
            if self.lazy_load:
                # Lazy load: only store path, load image on access
                sample = ImageTreePair(
                    image_id=image_id,
                    image_array=None,  # Will be loaded on access
                    distances=distances,
                    sequence_name=img_data.get("sequence_name"),
                    sequence_length=img_data.get("sequence_length"),
                )
                # Store path for lazy loading
                sample._image_path = str(image_path)
                sample._image_dataset_path = dataset_path
            else:
                # Eager load: load image immediately
                image_array = self._load_image(image_path, dataset_path)
                sample = ImageTreePair(
                    image_id=image_id,
                    image_array=image_array,
                    distances=distances,
                    sequence_name=img_data.get("sequence_name"),
                    sequence_length=img_data.get("sequence_length"),
                )
            
            self.samples.append(sample)
    
    @staticmethod
    def _load_image(image_path: Path, dataset_path: Optional[str] = None) -> np.ndarray:
        """Load image from HDF5 shards or legacy .npy files."""
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
                            "dataset_path is required"
                        )
                    target_dataset = dataset_names[0]

                if target_dataset not in handle:
                    raise FileNotFoundError(
                        f"Dataset '{target_dataset}' not found in HDF5 file {image_path}"
                    )
                return handle[target_dataset][...]

        return np.load(image_path)
    
    def _load_distances_for_image(
        self,
        distances_file: str,
        sequence_name: str,
    ) -> Optional[np.ndarray]:
        """
        Load distance vector for a specific sequence from distances CSV.
        
        Args:
            distances_file: Path to distances CSV
            sequence_name: Name of sequence to get distances for
            
        Returns:
            Distances vector or None if file not found
        """
        try:
            df = pd.read_csv(distances_file, index_col=0)
            if sequence_name in df.index:
                return df.loc[sequence_name].values.astype(np.float32)
        except (FileNotFoundError, KeyError, pd.errors.ParserError):
            pass
        
        return None
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> ImageTreePair:
        """
        Get sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            ImageTreePair with image and optional distances
        """
        sample = self.samples[index]
        
        # Lazy load image if needed
        if self.lazy_load and sample.image_array is None:
            raw_path = str(sample._image_path)
            if "images_output/images_output" in raw_path:
                raw_path = raw_path.replace("images_output/images_output", "images_output")
            corrected_path = Path(raw_path)
            sample.image_array = self._load_image(
                corrected_path,
                getattr(sample, "_image_dataset_path", None),
            )
        
        return sample
    
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        by_replicate: bool = True,
    ) -> Tuple["TrainingDataset", "TrainingDataset", "TrainingDataset"]:
        """
        Split dataset into train/val/test subsets.
        
        Args:
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation (default: 0.15)
            test_ratio: Fraction for testing (default: 0.15)
            by_replicate: Split by replicate ID to avoid data leakage (default: True)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):  # Allow small float error
            raise ValueError(f"Ratios must sum to 1, got {total}")
        
        if by_replicate:
            # Group by replicate ID (first part of image_id)
            replicates: Dict[str, List[int]] = {}
            for i, sample in enumerate(self.samples):
                # Extract replicate ID from image_id (e.g., "training_001_001" -> "training_001")
                parts = sample.image_id.split('_')
                if len(parts) >= 2:
                    rep_id = '_'.join(parts[:2])
                else:
                    rep_id = parts[0]
                
                if rep_id not in replicates:
                    replicates[rep_id] = []
                replicates[rep_id].append(i)
            
            # Split replicates
            rep_list = list(replicates.keys())
            n_reps = len(rep_list)
            train_end = int(n_reps * train_ratio)
            val_end = train_end + int(n_reps * val_ratio)
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            for i, rep_id in enumerate(rep_list):
                if i < train_end:
                    train_indices.extend(replicates[rep_id])
                elif i < val_end:
                    val_indices.extend(replicates[rep_id])
                else:
                    test_indices.extend(replicates[rep_id])
        else:
            # Simple random split
            n_samples = len(self.samples)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)
            
            indices = list(range(n_samples))
            np.random.shuffle(indices)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        
        # Create subset datasets
        train_dataset = self._subset(train_indices)
        val_dataset = self._subset(val_indices)
        test_dataset = self._subset(test_indices)
        
        print(f"Split dataset: train={len(train_dataset)}, "
              f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _subset(self, indices: List[int]) -> "TrainingDataset":
        """Create a subset dataset with specified indices."""
        subset = TrainingDataset.__new__(TrainingDataset)
        subset.root_dir = self.root_dir
        subset.lazy_load = self.lazy_load
        subset.manifest = self.manifest
        subset.distance_caches = self.distance_caches
        subset.samples = [self.samples[i] for i in indices]
        return subset
    
    def get_image_shape(self) -> Tuple[int, int, int]:
        """Get shape of images in this dataset."""
        if len(self.samples) > 0:
            sample = self[0]
            return sample.image_array.shape
        return (0, 0, 3)
    
    def get_distance_shape(self) -> Optional[Tuple[int,]]:
        """Get shape of distance vectors (if available)."""
        for sample in self.samples:
            if sample.distances is not None:
                return sample.distances.shape
        return None


class DatasetSplitter:
    """Utility for managing train/val/test splits."""
    
    @staticmethod
    def stratified_split(
        dataset: TrainingDataset,
        stratify_by: str = "replicate",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[TrainingDataset, TrainingDataset, TrainingDataset]:
        """
        Create stratified split by replicate or other grouping.
        
        Args:
            dataset: Input dataset
            stratify_by: Grouping strategy ('replicate', 'random')
            train_ratio: Training fraction
            val_ratio: Validation fraction
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        return dataset.split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            by_replicate=(stratify_by == "replicate"),
        )
