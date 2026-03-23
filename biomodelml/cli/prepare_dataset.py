import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
from tqdm import tqdm
import h5py
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from biomodelml.simulation import SyntheticEvolutionGenerator
from biomodelml.matrices import build_matrix

def process_and_save(args):
    """
    Generates a dataset of sequence pairs and saves them to disk using HDF5 for efficient I/O.
    """
    # 1. Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    # 2. Initialize the generator and transformation pipeline
    generator = SyntheticEvolutionGenerator(seq_len=args.seq_len)

    # Normalization parameters (will be applied during training)
    normalize_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    normalize_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    def process_sequence(seq):
        """Builds, pads, and transforms a single sequence matrix using PyTorch operations."""
        matrix = build_matrix(seq, seq, args.max_len, seq_type='N')
        h, w, c = matrix.shape

        # Convert to tensor first (HWC -> CHW format)
        tensor = torch.from_numpy(matrix).permute(2, 0, 1).float() / 255.0

        # Use PyTorch padding (more efficient than NumPy)
        # Pad format: (left, right, top, bottom)
        pad_h = args.max_len - h
        pad_w = args.max_len - w
        padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Apply normalization
        normalized_tensor = (padded_tensor - normalize_mean) / normalize_std

        return normalized_tensor

    print(f"Generating and saving {args.num_samples} samples to {args.output_dir}...")

    # 3. Create HDF5 file for efficient storage
    hdf5_path = os.path.join(args.output_dir, "dataset.h5")

    with h5py.File(hdf5_path, 'w') as h5f:
        # Pre-allocate datasets for efficient writing
        img1_dset = h5f.create_dataset('img1',
                                       shape=(args.num_samples, 3, args.max_len, args.max_len),
                                       dtype='float32',
                                       compression='gzip',
                                       compression_opts=4)
        img2_dset = h5f.create_dataset('img2',
                                       shape=(args.num_samples, 3, args.max_len, args.max_len),
                                       dtype='float32',
                                       compression='gzip',
                                       compression_opts=4)
        dist_dset = h5f.create_dataset('dist',
                                       shape=(args.num_samples, 1),
                                       dtype='float32')

        # Generation and Saving Loop
        for i in tqdm(range(args.num_samples), desc="Generating Dataset"):
            # Generate a synthetic evolution pair
            parent_seq, mutated_seq, distance = generator.generate_evolution_pair()

            # Process each sequence into a tensor
            img1_tensor = process_sequence(parent_seq.seq)
            img2_tensor = process_sequence(mutated_seq.seq)

            # Write directly to HDF5 (no intermediate file I/O)
            img1_dset[i] = img1_tensor.numpy()
            img2_dset[i] = img2_tensor.numpy()
            dist_dset[i] = distance

    print(f"Dataset generation complete. Saved to {hdf5_path}")
    print(f"File size: {os.path.getsize(hdf5_path) / (1024**3):.2f} GB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-generate and cache a dataset for the Siamese Regressor.")
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of synthetic pairs to generate.')
    parser.add_argument('--seq_len', type=int, default=500, help='Length of the sequences to simulate.')
    parser.add_argument('--max_len', type=int, default=550, help='Maximum length for padding image matrices.')
    parser.add_argument('--output_dir', type=str, default='data/siamese_cache', help='Directory to save the cached dataset.')

    args = parser.parse_args()
    process_and_save(args)
