#!/usr/bin/env python
"""Build phylogenetic trees using multiple similarity algorithms."""

import argparse
import os
import sys
from pathlib import Path
from biomodelml.experiment import Experiment
from biomodelml.variants.control import ControlVariant
from biomodelml.variants.sw import SmithWatermanVariant
from biomodelml.variants.nw import NeedlemanWunschVariant
from biomodelml.variants.uqi import UQIVariant
from biomodelml.variants.resized_ssim import ResizedSSIMVariant
from biomodelml.variants.resized_ssim_multiscale import ResizedSSIMMultiScaleVariant
from biomodelml.variants.windowed_ssim_multiscale import WindowedSSIMMultiScaleVariant
from biomodelml.variants.greedy_ssim import GreedySSIMVariant
from biomodelml.variants.unrestricted_ssim import UnrestrictedSSIMVariant
from biomodelml.variants.deep_search.variant import DeepSearchVariant


def build_trees(fasta_file: str, output_path: str, sequence_type: str, 
                image_path: str = None, algorithms: list = None):
    """
    Build phylogenetic trees using specified algorithms.
    
    Args:
        fasta_file: Path to sanitized FASTA file
        output_path: Directory to save results
        sequence_type: 'N' for nucleotides or 'P' for proteins
        image_path: Optional path to pre-generated matrix images
        algorithms: List of algorithm names to use (default: all)
    """
    # Default: run all algorithms
    all_variants = {
        'control': lambda: ControlVariant(fasta_file, sequence_type),
        'sw': lambda: SmithWatermanVariant(fasta_file, sequence_type),
        'nw': lambda: NeedlemanWunschVariant(fasta_file, sequence_type),
        'rssim': lambda: ResizedSSIMVariant(fasta_file, sequence_type, image_path),
        'rmsssim': lambda: ResizedSSIMMultiScaleVariant(fasta_file, sequence_type, image_path),
        'wmsssim': lambda: WindowedSSIMMultiScaleVariant(fasta_file, sequence_type, image_path),
        'gssim': lambda: GreedySSIMVariant(fasta_file, sequence_type, image_path),
        'ussim': lambda: UnrestrictedSSIMVariant(fasta_file, sequence_type, image_path),
        'uqi': lambda: UQIVariant(fasta_file, sequence_type, image_path),
        'deep': lambda: DeepSearchVariant(fasta_file, sequence_type, image_path)
    }
    
    # Select algorithms to run
    if algorithms:
        selected_variants = [all_variants[alg]() for alg in algorithms if alg in all_variants]
    else:
        selected_variants = [v() for v in all_variants.values()]
    
    if not selected_variants:
        raise ValueError("No valid algorithms selected")
    
    print(f"Running {len(selected_variants)} algorithms on {fasta_file}")
    
    # Run experiment
    Experiment(Path(output_path), *selected_variants).run_and_save()
    
    print(f"Results saved to {output_path}")


def main():
    """Build phylogenetic dendrograms using image-based similarity algorithms."""
    parser = argparse.ArgumentParser(
        description="Build phylogenetic trees using multiple similarity algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available algorithms:
  control   - Clustal Omega (requires external installation)
  sw        - Smith-Waterman
  nw        - Needleman-Wunsch
  rssim     - Resized SSIM
  rmsssim   - Resized MS-SSIM
  wmsssim   - Windowed MS-SSIM
  gssim     - Greedy Sliced SSIM
  ussim     - Unrestricted Sliced SSIM
  uqi       - Universal Quality Index
  deep      - Deep Search (VGG16 + Annoy)

Examples:
  # Run all algorithms
  %(prog)s sequences.fasta.N.sanitized output/ N
  
  # Run specific algorithms
  %(prog)s sequences.fasta.N.sanitized output/ N --algorithms rmsssim ussim
  
  # Use pre-generated images
  %(prog)s sequences.fasta.N.sanitized output/ N --image-path data/images/
        """
    )
    
    parser.add_argument(
        "fasta_file",
        help="Path to the sanitized FASTA file"
    )
    
    parser.add_argument(
        "output_path",
        help="Directory to save phylogenetic tree results"
    )
    
    parser.add_argument(
        "seq_type",
        choices=["N", "P"],
        help="Sequence type: N for nucleotides, P for proteins"
    )
    
    parser.add_argument(
        "--image-path",
        help="Path to pre-generated matrix images (optional)"
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=['control', 'sw', 'nw', 'rssim', 'rmsssim', 'wmsssim', 
                 'gssim', 'ussim', 'uqi', 'deep'],
        help="Specific algorithms to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate image path if provided
    if args.image_path and not os.path.exists(args.image_path):
        print(f"Warning: Image path {args.image_path} does not exist", file=sys.stderr)
    
    # Create output subdirectory
    base_name = args.fasta_file.split(".")[0].split("/")[-1]
    output_dir = os.path.join(args.output_path, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        build_trees(args.fasta_file, output_dir, args.seq_type, 
                   args.image_path, args.algorithms)
    except Exception as e:
        print(f"Error building trees: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
