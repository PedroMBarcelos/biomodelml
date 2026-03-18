#!/usr/bin/env python3
"""
Migration script to convert existing .pt file-based datasets to HDF5 format.

This script reads all .pt files from a cache directory and consolidates them
into a single HDF5 file for much faster I/O during training.
"""
import torch
import h5py
import os
import argparse
from tqdm import tqdm


def migrate_to_hdf5(input_dir, output_path=None):
    """
    Migrates a directory of .pt files to a single HDF5 file.

    Args:
        input_dir: Directory containing .pt files
        output_path: Optional output path for HDF5 file (defaults to input_dir/dataset.h5)
    """
    if output_path is None:
        output_path = os.path.join(input_dir, "dataset.h5")

    # Get all .pt files
    pt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pt')])
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return

    num_samples = len(pt_files)
    print(f"Found {num_samples} .pt files to migrate")

    # Load first sample to get dimensions
    first_sample = torch.load(os.path.join(input_dir, pt_files[0]))
    img_shape = first_sample['img1'].shape
    print(f"Sample image shape: {img_shape}")

    # Create HDF5 file
    print(f"Creating HDF5 file at {output_path}")
    with h5py.File(output_path, 'w') as h5f:
        # Pre-allocate datasets
        img1_dset = h5f.create_dataset('img1',
                                       shape=(num_samples,) + img_shape,
                                       dtype='float32',
                                       compression='gzip',
                                       compression_opts=4)
        img2_dset = h5f.create_dataset('img2',
                                       shape=(num_samples,) + img_shape,
                                       dtype='float32',
                                       compression='gzip',
                                       compression_opts=4)
        dist_dset = h5f.create_dataset('dist',
                                       shape=(num_samples, 1),
                                       dtype='float32')

        # Migrate each file
        for i, pt_file in enumerate(tqdm(pt_files, desc="Migrating to HDF5")):
            file_path = os.path.join(input_dir, pt_file)
            data = torch.load(file_path)

            img1_dset[i] = data['img1'].numpy()
            img2_dset[i] = data['img2'].numpy()
            dist_dset[i] = data['dist'].numpy()

    print(f"\nMigration complete!")
    print(f"HDF5 file size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    # Calculate space savings
    pt_total_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in pt_files)
    hdf5_size = os.path.getsize(output_path)
    savings = (1 - hdf5_size / pt_total_size) * 100

    print(f"Original .pt files total: {pt_total_size / (1024**3):.2f} GB")
    print(f"New HDF5 file: {hdf5_size / (1024**3):.2f} GB")
    print(f"Space savings: {savings:.1f}%")
    print(f"\nYou can now delete the .pt files and use {output_path} for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Migrate .pt file dataset to HDF5 format")
    parser.add_argument('input_dir', type=str, help='Directory containing .pt files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for HDF5 file (default: input_dir/dataset.h5)')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist")
        exit(1)

    migrate_to_hdf5(args.input_dir, args.output)
