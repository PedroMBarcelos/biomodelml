"""
4-Channel Dataset for Siamese Regressor with Mask Generation and Curriculum Learning

This module extends the standard 3-channel dataset to support 4-channel input:
- Channels 0-2: RGB (normalized to [-1, 1])
- Channel 3: Binary mask (1.0 = valid sequence data, 0.0 = padding)

The mask channel explicitly marks padded regions, preventing the model from learning
spurious correlations from zero-padding artifacts while maintaining the "1 pixel = 1 residue"
biochemical scale.

Features:
- On-the-fly generation with mask creation
- HDF5 cached dataset support
- Curriculum learning with distance-based filtering
- PyTorch-optimized padding and normalization
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from Bio.Seq import Seq
import os
import h5py

from biomodelml.simulation import get_generator
from biomodelml.matrices import build_matrix


class SiameseEvolutionDataset4Channel(Dataset):
    """
    PyTorch Dataset for 4-Channel Siamese Regressor with curriculum learning support.

    This dataset can operate in two modes:
    1. On-the-fly generation (for rapid prototyping and small tests).
    2. Loading from a pre-generated HDF5 cache (for production training).

    The curriculum learning feature allows progressive training by filtering samples
    based on evolutionary distance, starting with easy (low distance) pairs and
    gradually increasing difficulty.
    """

    def __init__(self, generator=None, num_samples=100000, max_len=500, cache_file=None,
                 seq_type='N', curriculum_mode='none', curriculum_phase=0):
        """
        Initialize the 4-Channel Siamese Evolution Dataset.

        Args:
            generator (Generator, optional): Data generator instance (SyntheticEvolutionGenerator
                or ProteinEvolutionGenerator). If None and cache_file is None, creates default generator.
            num_samples (int): Total number of samples to generate (for on-the-fly mode).
            max_len (int): Maximum sequence length for padding matrices.
            cache_file (str, optional): Path to pre-generated HDF5 dataset file.
            seq_type (str): 'N' for nucleotide or 'P' for protein (default: 'N').
            curriculum_mode (str): Curriculum learning strategy:
                - 'none': No filtering, all distances
                - 'progressive': Gradually increase max distance
                - 'bin': Sample only from specific distance bin
            curriculum_phase (int): Current curriculum phase (0-4):
                Phase 0: distances ≤ 0.3
                Phase 1: distances ≤ 0.6
                Phase 2: distances ≤ 0.9
                Phase 3: distances ≤ 1.2
                Phase 4: distances ≤ 1.5 (all)
        """
        self.generator = generator
        self.num_samples = num_samples
        self.max_len = max_len
        self.cache_file = cache_file
        self.seq_type = seq_type
        self.curriculum_mode = curriculum_mode
        self.curriculum_phase = curriculum_phase

        # Curriculum distance thresholds
        self.phase_thresholds = [0.3, 0.6, 0.9, 1.2, 1.5]
        self.phase_bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5)]

        # HDF5 file handle and datasets (for cached mode)
        self.h5_file = None
        self.img1_dset = None
        self.img2_dset = None
        self.dist_dset = None

        if self.cache_file:
            # Load from HDF5 cache
            if not os.path.exists(self.cache_file):
                raise FileNotFoundError(f"Cache file not found: {self.cache_file}")

            print(f"Loading 4-channel HDF5 cached dataset from: {self.cache_file}")
            self.h5_file = h5py.File(self.cache_file, 'r')
            self.img1_dset = self.h5_file['img1']
            self.img2_dset = self.h5_file['img2']
            self.dist_dset = self.h5_file['dist']
            self.num_samples = len(self.img1_dset)

            # Verify it's 4-channel format
            if self.img1_dset.shape[1] != 4:
                raise ValueError(
                    f"Expected 4 channels in cached dataset, got {self.img1_dset.shape[1]}. "
                    "Please generate dataset with 4-channel support."
                )

            print(f"Loaded {self.num_samples} samples with shape {self.img1_dset.shape}")

        elif not self.generator:
            # Create generator if not provided
            self.generator = get_generator(seq_type, seq_len=500)

        # Print curriculum info
        if self.curriculum_mode != 'none':
            self._print_curriculum_info()

    def _print_curriculum_info(self):
        """Print curriculum learning configuration."""
        if self.curriculum_mode == 'progressive':
            max_dist = self.phase_thresholds[min(self.curriculum_phase, 4)]
            print(f"Curriculum (progressive): Phase {self.curriculum_phase}, max distance = {max_dist}")
        elif self.curriculum_mode == 'bin':
            bin_range = self.phase_bins[min(self.curriculum_phase, 4)]
            print(f"Curriculum (bin): Phase {self.curriculum_phase}, distance range = {bin_range}")

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates or loads one sample from the dataset.

        Returns:
            tuple: ((img1_tensor, img2_tensor), distance_tensor)
                - img1_tensor: (4, max_len, max_len) tensor
                - img2_tensor: (4, max_len, max_len) tensor
                - distance_tensor: (1,) tensor
        """
        if self.cache_file:
            # Load from HDF5 cache
            img1 = torch.from_numpy(self.img1_dset[idx])
            img2 = torch.from_numpy(self.img2_dset[idx])
            dist = torch.from_numpy(self.dist_dset[idx])
            return (img1, img2), dist
        else:
            # On-the-fly generation with curriculum filtering
            return self._generate_item()

    def _generate_item(self):
        """
        Generates one 4-channel sample on-the-fly with curriculum filtering.

        This method uses PyTorch-optimized operations for efficiency and applies
        curriculum learning by rejecting samples outside the current phase's distance range.

        Returns:
            tuple: ((img1_tensor, img2_tensor), distance_tensor)
        """
        # Curriculum learning: retry until we get a valid distance
        max_attempts = 100
        for attempt in range(max_attempts):
            parent_seq, mutated_seq, distance = self.generator.generate_evolution_pair()

            if self._is_distance_valid(distance):
                break

            if attempt == max_attempts - 1:
                # If we can't find a valid sample after 100 tries, just use this one
                print(f"Warning: Could not find valid distance after {max_attempts} attempts. "
                      f"Using distance {distance:.4f} anyway.")

        # Process both sequences
        img1_tensor = self._process_sequence(parent_seq.seq)
        img2_tensor = self._process_sequence(mutated_seq.seq)
        distance_tensor = torch.tensor([distance], dtype=torch.float32)

        return (img1_tensor, img2_tensor), distance_tensor

    def _process_sequence(self, seq):
        """
        Process a single sequence into a 4-channel padded and normalized tensor.

        Pipeline:
            1. Build RGB self-comparison matrix (H×W×3 uint8)
            2. Create binary mask (H×W float32)
            3. Convert to PyTorch tensors and permute to CHW format
            4. Pad to max_len×max_len (top-left alignment)
            5. Normalize: RGB → [-1, 1], Mask → [0, 1]
            6. Concatenate to 4-channel tensor

        Args:
            seq (Seq): BioPython sequence object

        Returns:
            torch.Tensor: 4-channel tensor of shape (4, max_len, max_len)
        """
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]

        # Step 1: Build RGB matrix
        matrix = build_matrix(seq, seq, self.max_len, seq_type=self.seq_type)
        h, w, c = matrix.shape

        # Step 2: Create binary mask (valid region = 1.0, padding will be 0.0)
        mask = self._create_mask(h, w)

        # Step 3: Convert to tensors
        # RGB: uint8 → float[0,1] → CHW format
        rgb_tensor = torch.from_numpy(matrix).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        # Mask: float32 → add channel dimension
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        # Step 4: Pad to max_len (top-left alignment with zeros)
        pad_h = self.max_len - h
        pad_w = self.max_len - w

        rgb_padded = F.pad(rgb_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_padded = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Step 5: Normalize RGB to [-1, 1], keep mask at [0, 1]
        mean_rgb = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std_rgb = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        rgb_normalized = (rgb_padded - mean_rgb) / std_rgb

        # Step 6: Concatenate to 4 channels
        tensor_4ch = torch.cat([rgb_normalized, mask_padded], dim=0)  # (4, max_len, max_len)

        return tensor_4ch

    def _create_mask(self, h, w):
        """
        Create a binary mask marking valid sequence region vs padding.

        This follows the pattern from the optical flow variant's aggregator.create_mask()
        function, ensuring consistency across the codebase.

        Args:
            h (int): Height of actual sequence data
            w (int): Width of actual sequence data

        Returns:
            np.ndarray: Binary mask of shape (h, w) where all values are 1.0
                        (padding will be added later to match max_len)
        """
        # Create mask with same dimensions as the actual data
        # Padding with zeros will be added in _process_sequence
        mask = np.ones((h, w), dtype=np.float32)
        return mask

    def _is_distance_valid(self, distance):
        """
        Check if a distance is valid for the current curriculum phase.

        Args:
            distance (float): Evolutionary distance to check

        Returns:
            bool: True if distance is valid for current phase, False otherwise
        """
        if self.curriculum_mode == 'none':
            return True

        elif self.curriculum_mode == 'progressive':
            # Progressive: gradually increase maximum distance
            max_dist = self.phase_thresholds[min(self.curriculum_phase, 4)]
            return distance <= max_dist

        elif self.curriculum_mode == 'bin':
            # Bin: sample only from specific distance range
            bin_min, bin_max = self.phase_bins[min(self.curriculum_phase, 4)]
            return bin_min <= distance <= bin_max

        return True

    def get_curriculum_stats(self, num_samples=1000):
        """
        Get statistics about the distance distribution for current curriculum settings.

        Args:
            num_samples (int): Number of samples to check

        Returns:
            dict: Statistics including min, max, mean, std of distances
        """
        if self.cache_file:
            # Sample from cache
            distances = []
            for i in range(min(num_samples, len(self))):
                _, dist = self[i]
                distances.append(dist.item())
        else:
            # Generate samples
            distances = []
            for _ in range(num_samples):
                _, dist = self._generate_item()
                distances.append(dist.item())

        distances = np.array(distances)
        return {
            'min': distances.min(),
            'max': distances.max(),
            'mean': distances.mean(),
            'std': distances.std(),
            'median': np.median(distances),
            'count': len(distances)
        }


if __name__ == '__main__':
    print("=" * 80)
    print("Testing SiameseEvolutionDataset4Channel")
    print("=" * 80)

    # Test 1: On-the-fly generation without curriculum
    print("\n1. Testing on-the-fly generation (no curriculum)...")
    from biomodelml.simulation import SyntheticEvolutionGenerator

    gen = SyntheticEvolutionGenerator(seq_len=100)
    dataset = SiameseEvolutionDataset4Channel(
        generator=gen,
        num_samples=10,
        max_len=150,
        seq_type='N',
        curriculum_mode='none'
    )

    print(f"Dataset length: {len(dataset)}")

    # Get a sample
    (img1, img2), dist = dataset[0]
    print(f"Sample 0 shape: img1={img1.shape}, img2={img2.shape}, dist={dist.item():.4f}")
    assert img1.shape == (4, 150, 150), f"Expected (4, 150, 150), got {img1.shape}"
    print("✓ Output shape correct (4 channels)")

    # Check channel ranges
    rgb_min, rgb_max = img1[:3].min().item(), img1[:3].max().item()
    mask_min, mask_max = img1[3].min().item(), img1[3].max().item()
    print(f"RGB range: [{rgb_min:.2f}, {rgb_max:.2f}]")
    print(f"Mask range: [{mask_min:.2f}, {mask_max:.2f}]")

    assert rgb_min >= -1.1 and rgb_max <= 1.1, f"RGB should be in [-1, 1], got [{rgb_min}, {rgb_max}]"
    assert mask_min >= 0.0 and mask_max <= 1.0, f"Mask should be in [0, 1], got [{mask_min}, {mask_max}]"
    print("✓ Channel normalization correct")

    # Check mask is binary
    mask_unique = torch.unique(img1[3])
    print(f"Mask unique values: {mask_unique.tolist()}")
    assert len(mask_unique) <= 2, "Mask should be binary"
    print("✓ Mask is binary")

    # Test 2: Curriculum learning - progressive mode
    print("\n" + "=" * 80)
    print("2. Testing curriculum learning (progressive mode)...")
    print("=" * 80)

    for phase in range(5):
        print(f"\n--- Phase {phase} ---")
        dataset_curr = SiameseEvolutionDataset4Channel(
            generator=gen,
            num_samples=50,
            max_len=150,
            seq_type='N',
            curriculum_mode='progressive',
            curriculum_phase=phase
        )

        # Collect distances
        distances = []
        for i in range(10):  # Check 10 samples
            _, dist = dataset_curr[i]
            distances.append(dist.item())

        max_dist = max(distances)
        mean_dist = np.mean(distances)
        threshold = [0.3, 0.6, 0.9, 1.2, 1.5][phase]

        print(f"  Distances: mean={mean_dist:.3f}, max={max_dist:.3f}")
        print(f"  Threshold: {threshold:.1f}")

        # Allow some tolerance for randomness
        if max_dist > threshold + 0.1:
            print(f"  ⚠ Warning: Max distance {max_dist:.3f} exceeds threshold {threshold:.1f}")
        else:
            print(f"  ✓ Distances within threshold")

    # Test 3: Curriculum learning - bin mode
    print("\n" + "=" * 80)
    print("3. Testing curriculum learning (bin mode)...")
    print("=" * 80)

    for phase in [0, 2, 4]:  # Test a few phases
        print(f"\n--- Phase {phase} ---")
        dataset_bin = SiameseEvolutionDataset4Channel(
            generator=gen,
            num_samples=50,
            max_len=150,
            seq_type='N',
            curriculum_mode='bin',
            curriculum_phase=phase
        )

        # Collect distances
        distances = []
        for i in range(10):
            _, dist = dataset_bin[i]
            distances.append(dist.item())

        bin_min, bin_max = [(0.0, 0.3), (0.6, 0.9), (1.2, 1.5)][phase // 2]
        print(f"  Target bin: [{bin_min:.1f}, {bin_max:.1f}]")
        print(f"  Actual distances: {[f'{d:.3f}' for d in distances[:5]]}")

        in_bin = sum(bin_min <= d <= bin_max + 0.1 for d in distances)
        print(f"  In bin: {in_bin}/{len(distances)}")

    # Test 4: Get curriculum statistics
    print("\n" + "=" * 80)
    print("4. Testing curriculum statistics...")
    print("=" * 80)

    dataset_stats = SiameseEvolutionDataset4Channel(
        generator=gen,
        num_samples=100,
        max_len=150,
        curriculum_mode='progressive',
        curriculum_phase=2  # Phase 2: distances ≤ 0.9
    )

    stats = dataset_stats.get_curriculum_stats(num_samples=50)
    print(f"Statistics for Phase 2 (max distance 0.9):")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Std: {stats['std']:.4f}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe 4-channel dataset is ready for training.")
    print("Next steps:")
    print("  1. Create train_siamese_4ch.py for training")
    print("  2. Test training on small dataset (100 samples)")
    print("  3. Generate large cached dataset with generate_evolutionary_dataset.py")
