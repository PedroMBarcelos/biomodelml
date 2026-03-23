import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from Bio.Seq import Seq
import os
import h5py


from biomodelml.simulation import get_generator
from biomodelml.matrices import build_matrix

class SiameseEvolutionDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Regressor.
    This dataset can operate in two modes:
    1. On-the-fly generation (for small tests).
    2. Loading from a pre-generated HDF5 cache (for training).
    """
    def __init__(self, generator=None, num_samples=100000, max_len=500, transform=None, cache_dir=None, seq_type='N'):
        """
        Args:
            generator (Generator, optional): The data generator instance.
            num_samples (int): The total number of samples.
            max_len (int): The maximum sequence length for padding matrices.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_dir (str, optional): Directory to load pre-generated HDF5 dataset from.
            seq_type (str): 'N' for nucleotide or 'P' for protein (default: 'N').
        """
        self.generator = generator
        self.num_samples = num_samples
        self.max_len = max_len
        self.transform = transform
        self.cache_dir = cache_dir
        self.seq_type = seq_type
        self.h5_file = None
        self.img1_dset = None
        self.img2_dset = None
        self.dist_dset = None

        if self.cache_dir:
            # Check for new format: dataset_{seq_type}.h5
            seq_type_name = 'nucleotide' if seq_type == 'N' else 'protein'
            hdf5_path_new = os.path.join(self.cache_dir, f"dataset_{seq_type_name}.h5")
            # Check for old format: dataset.h5 (backward compatibility)
            hdf5_path_old = os.path.join(self.cache_dir, "dataset.h5")

            if os.path.exists(hdf5_path_new):
                print(f"Using HDF5 cached dataset from: {hdf5_path_new}")
                self.h5_file = h5py.File(hdf5_path_new, 'r')
                self.img1_dset = self.h5_file['img1']
                self.img2_dset = self.h5_file['img2']
                self.dist_dset = self.h5_file['dist']
                self.num_samples = len(self.img1_dset)
            elif os.path.exists(hdf5_path_old):
                # Backward compatibility: use old format
                print(f"Using HDF5 cached dataset from: {hdf5_path_old} (legacy format)")
                self.h5_file = h5py.File(hdf5_path_old, 'r')
                self.img1_dset = self.h5_file['img1']
                self.img2_dset = self.h5_file['img2']
                self.dist_dset = self.h5_file['dist']
                self.num_samples = len(self.img1_dset)
            else:
                # Fallback to old .pt files if HDF5 doesn't exist
                print(f"Warning: HDF5 file not found. Falling back to .pt files from: {self.cache_dir}")
                self.file_list = sorted([os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith('.pt')])
                self.num_samples = len(self.file_list)
        elif not self.generator:
            # Create generator if not provided
            self.generator = get_generator(seq_type, seq_len=500)

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()

    def _get_default_transform(self):
        """
        Returns a default transformation pipeline for the image matrices.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates or loads one sample from the dataset.
        """
        if self.cache_dir:
            if self.h5_file is not None:
                # Load from HDF5 (memory-mapped, very fast)
                img1 = torch.from_numpy(self.img1_dset[idx])
                img2 = torch.from_numpy(self.img2_dset[idx])
                dist = torch.from_numpy(self.dist_dset[idx])
                return (img1, img2), dist
            else:
                # Fallback to .pt files
                file_path = self.file_list[idx]
                data = torch.load(file_path)
                return (data['img1'], data['img2']), data['dist']
        else:
            # On-the-fly generation (memory-optimized)
            return self._generate_item()

    def _generate_item(self):
        """
        Generates one sample on-the-fly using PyTorch padding.
        """
        parent_seq, mutated_seq, distance = self.generator.generate_evolution_pair()

        def process_sequence(seq):
            matrix = build_matrix(seq, seq, self.max_len, seq_type=self.seq_type)
            h, w, c = matrix.shape

            # Convert to tensor first (HWC -> CHW format)
            tensor = torch.from_numpy(matrix).permute(2, 0, 1).float() / 255.0

            # Use PyTorch padding (more efficient than NumPy)
            pad_h = self.max_len - h
            pad_w = self.max_len - w
            padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

            # Apply normalization
            if self.transform:
                return self.transform(padded_tensor)
            else:
                # Default normalization
                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                return (padded_tensor - mean) / std

        img1_tensor = process_sequence(parent_seq.seq)
        img2_tensor = process_sequence(mutated_seq.seq)
        distance_tensor = torch.tensor([distance], dtype=torch.float32)

        return (img1_tensor, img2_tensor), distance_tensor

if __name__ == '__main__':
    # Example of how to use the Dataset with on-the-fly generation
    print("--- Testing On-the-fly Generation ---")
    gen = SyntheticEvolutionGenerator(seq_len=100)
    dataset_live = SiameseEvolutionDataset(generator=gen, num_samples=10, max_len=150)
    print(f"Live dataset length: {len(dataset_live)}")
    (img1, img2), dist = dataset_live[0]
    print(f"Sample 0 - Image 1 Tensor Shape: {img1.shape}, Distance: {dist.item():.4f}")

    # Example of how to use the Dataset with a cache
    # First, create a dummy cache
    print("\n--- Testing Cached Dataset ---")
    dummy_cache = 'data/dummy_cache'
    if not os.path.exists(dummy_cache):
        os.makedirs(dummy_cache)
    for i in range(5):
        torch.save({
            'img1': torch.randn(3, 150, 150),
            'img2': torch.randn(3, 150, 150),
            'dist': torch.tensor([0.1 * i])
        }, os.path.join(dummy_cache, f'sample_{i}.pt'))
    
    dataset_cached = SiameseEvolutionDataset(cache_dir=dummy_cache)
    print(f"Cached dataset length: {len(dataset_cached)}")
    (img1_c, img2_c), dist_c = dataset_cached[0]
    print(f"Sample 0 - Image 1 Tensor Shape: {img1_c.shape}, Distance: {dist_c.item():.4f}")
