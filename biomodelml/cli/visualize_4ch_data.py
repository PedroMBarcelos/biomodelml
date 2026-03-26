"""
Visualization Script for 4-Channel Siamese Dataset

This script visualizes the matrices generated for training:
- RGB channels (sequence comparison matrix)
- Mask channel (valid data vs padding)
- Side-by-side comparison of image pairs
- Distance information

Usage:
    # Visualize from on-the-fly generation
    python -m biomodelml.cli.visualize_4ch_data --num-samples 5

    # Visualize from cached HDF5 dataset
    python -m biomodelml.cli.visualize_4ch_data --cache-file data/evolutionary_10k.h5 --num-samples 5
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomodelml.datasets_4ch import SiameseEvolutionDataset4Channel
from biomodelml.simulation import get_generator


def denormalize_rgb(tensor):
    """
    Denormalize RGB channels from [-1, 1] back to [0, 1] for visualization.

    Args:
        tensor: RGB tensor in [-1, 1] range

    Returns:
        RGB tensor in [0, 1] range
    """
    return (tensor * 0.5) + 0.5


def visualize_4ch_sample(img1, img2, distance, sample_idx, output_dir):
    """
    Visualize a single 4-channel sample pair.

    Args:
        img1: First image tensor (4, H, W)
        img2: Second image tensor (4, H, W)
        distance: Evolutionary distance
        sample_idx: Sample index
        output_dir: Directory to save visualization
    """
    # Convert to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.numpy()
    if torch.is_tensor(img2):
        img2.numpy()

    # Extract channels
    rgb1 = img1[:3]  # (3, H, W)
    mask1 = img1[3]  # (H, W)
    rgb2 = img2[:3]
    mask2 = img2[3]

    # Denormalize RGB for visualization
    rgb1_vis = denormalize_rgb(rgb1)
    rgb2_vis = denormalize_rgb(rgb2)

    # Transpose to (H, W, 3) for matplotlib
    rgb1_vis = np.transpose(rgb1_vis, (1, 2, 0))
    rgb2_vis = np.transpose(rgb2_vis, (1, 2, 0))

    # Clip to [0, 1] for safety
    rgb1_vis = np.clip(rgb1_vis, 0, 1)
    rgb2_vis = np.clip(rgb2_vis, 0, 1)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Sample {sample_idx}: Evolutionary Distance = {distance:.4f}',
                 fontsize=16, fontweight='bold')

    # Row 1: RGB matrices
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb1_vis)
    ax1.set_title('Sequence 1: RGB Comparison Matrix', fontsize=12)
    ax1.set_xlabel('Position in Sequence 1')
    ax1.set_ylabel('Position in Sequence 1')
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb2_vis)
    ax2.set_title('Sequence 2: RGB Comparison Matrix', fontsize=12)
    ax2.set_xlabel('Position in Sequence 2')
    ax2.set_ylabel('Position in Sequence 2')
    ax2.grid(False)

    # Row 2: Mask channels
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(mask1, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Sequence 1: Mask Channel (1=valid, 0=padding)', fontsize=12)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Position')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.grid(False)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(mask2, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Sequence 2: Mask Channel (1=valid, 0=padding)', fontsize=12)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.grid(False)

    # Row 3: Individual RGB channels for Sequence 1
    ax5 = fig.add_subplot(gs[2, 0])
    # Stack RGB channels horizontally for comparison
    rgb1_channels = np.hstack([rgb1_vis[:, :, i] for i in range(3)])
    ax5.imshow(rgb1_channels, cmap='gray', vmin=0, vmax=1)
    ax5.set_title('Sequence 1: R | G | B Channels', fontsize=12)
    ax5.set_xticks([rgb1_vis.shape[1]//2, rgb1_vis.shape[1]*3//2, rgb1_vis.shape[1]*5//2])
    ax5.set_xticklabels(['Red', 'Green', 'Blue'])
    ax5.set_ylabel('Position')
    ax5.grid(False)

    # Row 3: Individual RGB channels for Sequence 2
    ax6 = fig.add_subplot(gs[2, 1])
    rgb2_channels = np.hstack([rgb2_vis[:, :, i] for i in range(3)])
    ax6.imshow(rgb2_channels, cmap='gray', vmin=0, vmax=1)
    ax6.set_title('Sequence 2: R | G | B Channels', fontsize=12)
    ax6.set_xticks([rgb2_vis.shape[1]//2, rgb2_vis.shape[1]*3//2, rgb2_vis.shape[1]*5//2])
    ax6.set_xticklabels(['Red', 'Green', 'Blue'])
    ax6.set_ylabel('Position')
    ax6.grid(False)

    # Add text annotation with statistics
    stats_text = (
        f"Sequence 1 Size: {int(mask1.sum()**0.5)}×{int(mask1.sum()**0.5)} bp\n"
        f"Sequence 2 Size: {int(mask2.sum()**0.5)}×{int(mask2.sum()**0.5)} bp\n"
        f"Padding: {(1 - mask1.mean())*100:.1f}% (Seq1), {(1 - mask2.mean())*100:.1f}% (Seq2)\n"
        f"RGB Range: [{rgb1_vis.min():.2f}, {rgb1_vis.max():.2f}]\n"
        f"Distance: {distance:.4f}"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    output_path = output_dir / f'sample_{sample_idx:03d}_dist_{distance:.4f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def visualize_distance_distribution(dataset, num_samples, output_dir):
    """
    Visualize the distribution of evolutionary distances in the dataset.

    Args:
        dataset: SiameseEvolutionDataset4Channel instance
        num_samples: Number of samples to analyze
        output_dir: Directory to save visualization
    """
    print(f"\nAnalyzing distance distribution ({num_samples} samples)...")

    distances = []
    for i in range(min(num_samples, len(dataset))):
        _, dist = dataset[i]
        distances.append(dist.item())

    distances = np.array(distances)

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(distances, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Evolutionary Distance', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distance Distribution ({len(distances)} samples)', fontsize=14)
    ax1.axvline(distances.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances.mean():.3f}')
    ax1.axvline(np.median(distances), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(distances, vert=True)
    ax2.set_ylabel('Evolutionary Distance', fontsize=12)
    ax2.set_title('Distance Box Plot', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = (
        f"Statistics:\n"
        f"  Count: {len(distances)}\n"
        f"  Min:    {distances.min():.4f}\n"
        f"  Max:    {distances.max():.4f}\n"
        f"  Mean:   {distances.mean():.4f}\n"
        f"  Median: {np.median(distances):.4f}\n"
        f"  Std:    {distances.std():.4f}"
    )
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'distance_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    print(f"\n  Min: {distances.min():.4f}, Max: {distances.max():.4f}, Mean: {distances.mean():.4f}")


def main(args):
    """Main visualization function."""

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("4-Channel Dataset Visualization")
    print("=" * 70)

    # Initialize dataset
    if args.cache_file:
        print(f"Loading dataset from: {args.cache_file}")
        dataset = SiameseEvolutionDataset4Channel(
            cache_file=args.cache_file,
            max_len=args.max_len
        )
    else:
        print("Generating samples on-the-fly...")
        generator = get_generator(args.seq_type, seq_len=args.seq_len)
        dataset = SiameseEvolutionDataset4Channel(
            generator=generator,
            num_samples=args.num_samples * 2,  # Generate extra for variety
            max_len=args.max_len,
            seq_type=args.seq_type
        )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Visualizing: {args.num_samples} samples")
    print(f"Output directory: {output_dir}")
    print()

    # Visualize individual samples
    print("Generating visualizations...")
    for i in range(min(args.num_samples, len(dataset))):
        (img1, img2), dist = dataset[i]
        visualize_4ch_sample(img1, img2, dist.item(), i, output_dir)

    # Visualize distance distribution
    if args.show_distribution:
        visualize_distance_distribution(dataset, min(args.num_samples * 10, len(dataset)), output_dir)

    print("\n" + "=" * 70)
    print(f"Visualization complete! Check {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize 4-channel Siamese dataset matrices"
    )

    # Dataset source
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Path to HDF5 cached dataset (leave empty for on-the-fly generation)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')

    # On-the-fly generation parameters (used if no cache-file)
    parser.add_argument('--seq-len', type=int, default=100,
                        help='Sequence length for on-the-fly generation')
    parser.add_argument('--max-len', type=int, default=150,
                        help='Matrix padding size')
    parser.add_argument('--seq-type', type=str, default='N', choices=['N', 'P'],
                        help='Sequence type')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--show-distribution', action='store_true',
                        help='Also create distance distribution plot')

    args = parser.parse_args()

    # Validate
    if args.cache_file and not Path(args.cache_file).exists():
        print(f"Error: Cache file not found: {args.cache_file}")
        sys.exit(1)

    main(args)
