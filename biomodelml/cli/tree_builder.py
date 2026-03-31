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
from biomodelml.variants.optical_flow import OpticalFlowVariant
from biomodelml.variants.siamese import SiameseVariant
from biomodelml.variants.baselines import (
    PDistanceVariant,
    JukesCantorVariant,
    GTRMLEVariant,
    MuscleIdentityVariant,
)


def build_trees(fasta_file: str, output_path: str, sequence_type: str, 
                image_path: str = None, algorithms: list = None,
                optflow_mode: str = "legacy",
                optflow_threshold: float = None,
                optflow_diagonal_width: int = None,
                optflow_highpass: bool = None,
                model_path: str = 'models/siamese_regressor.pth'):
    """
    Build phylogenetic trees using specified algorithms.
    
    Args:
        fasta_file: Path to sanitized FASTA file
        output_path: Directory to save results
        sequence_type: 'N' for nucleotides or 'P' for proteins
        image_path: Optional path to pre-generated matrix images
        algorithms: List of algorithm names to use (default: all)
        model_path: Path to the trained Siamese model
    """
    # Default: run all algorithms
    optflow_kwargs = {'optflow_mode': optflow_mode}
    if optflow_threshold is not None:
        optflow_kwargs['magnitude_threshold'] = optflow_threshold
    if optflow_diagonal_width is not None:
        optflow_kwargs['diagonal_ribbon_width'] = optflow_diagonal_width
    if optflow_highpass is not None:
        optflow_kwargs['highpass_enabled'] = optflow_highpass

    all_variants = {
        'control': lambda: ControlVariant(fasta_file, sequence_type),
        'pdist': lambda: PDistanceVariant(fasta_file, sequence_type),
        'jc69': lambda: JukesCantorVariant(fasta_file, sequence_type),
        'gtr_mle': lambda: GTRMLEVariant(fasta_file, sequence_type),
        'muscle': lambda: MuscleIdentityVariant(fasta_file, sequence_type),
        'sw': lambda: SmithWatermanVariant(fasta_file, sequence_type),
        'nw': lambda: NeedlemanWunschVariant(fasta_file, sequence_type),
        'rssim': lambda: ResizedSSIMVariant(fasta_file, sequence_type, image_path),
        'rmsssim': lambda: ResizedSSIMMultiScaleVariant(fasta_file, sequence_type, image_path),
        'wmsssim': lambda: WindowedSSIMMultiScaleVariant(fasta_file, sequence_type, image_path),
        'gssim': lambda: GreedySSIMVariant(fasta_file, sequence_type, image_path),
        'ussim': lambda: UnrestrictedSSIMVariant(fasta_file, sequence_type, image_path),
        'uqi': lambda: UQIVariant(fasta_file, sequence_type, image_path),
        'deep': lambda: DeepSearchVariant(fasta_file, sequence_type, image_path),
        'optflow': lambda: OpticalFlowVariant(fasta_file, sequence_type, image_path, **optflow_kwargs),
        'siamese': lambda: SiameseVariant(fasta_file, sequence_type, model_path=model_path)
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
    """Build phylogenetic dendrograms using baseline and image-based algorithms."""
    algorithm_choices = [
        'control', 'pdist', 'jc69', 'gtr_mle', 'muscle',
        'sw', 'nw', 'rssim', 'rmsssim', 'wmsssim',
        'gssim', 'ussim', 'uqi', 'deep', 'optflow', 'siamese'
    ]
    parser = argparse.ArgumentParser(
        description="Build phylogenetic trees using multiple similarity algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available algorithms:
  control   - Clustal Omega identity distance
  pdist     - p-distance over Clustal Omega MSA
  jc69      - Jukes-Cantor (JC69) over Clustal Omega MSA
  gtr_mle   - Pairwise GTR maximum-likelihood distance over Clustal Omega MSA
  muscle    - MUSCLE MSA + p-distance
  sw        - Smith-Waterman
  nw        - Needleman-Wunsch
  rssim     - Resized SSIM
  rmsssim   - Resized MS-SSIM
  wmsssim   - Windowed MS-SSIM
  gssim     - Greedy Sliced SSIM
  ussim     - Unrestricted Sliced SSIM
  uqi       - Universal Quality Index
  deep      - Deep Search (VGG16 + Annoy)
  optflow   - Dense Optical Flow with Farneback
  siamese   - Siamese Neural Network Regressor

Examples:
  # Run all algorithms
  %(prog)s sequences.fasta.N.sanitized output/ N

  # Run scientific baselines only
  %(prog)s sequences.fasta.N.sanitized output/ N --algorithms pdist jc69 gtr_mle muscle
        """
    )

    parser.add_argument("fasta_file", help="Path to the sanitized FASTA file")
    parser.add_argument("output_path", help="Directory to save phylogenetic tree results")
    parser.add_argument("seq_type", choices=["N", "P"], help="Sequence type: N for nucleotides, P for proteins")
    parser.add_argument("--image-path", help="Path to pre-generated matrix images (optional)")
    parser.add_argument("--algorithms", nargs="+", choices=algorithm_choices, help="Specific algorithms to run (default: all)")
    parser.add_argument(
        "--model-path",
        default="models/siamese_regressor.pth",
        help="Path to the trained Siamese model weights (for 'siamese' algorithm)"
    )
    parser.add_argument(
        "--optflow-mode",
        choices=["legacy", "strict"],
        default="legacy",
        help="Optical flow behavior preset: legacy (historical) or strict (aggressive denoising)"
    )
    parser.add_argument(
        "--optflow-threshold",
        type=float,
        default=None,
        help="Override optical flow magnitude threshold (e.g., 0.5 or 1.0)"
    )
    parser.add_argument(
        "--optflow-diagonal-width",
        type=int,
        default=None,
        help="Override diagonal ribbon width in pixels"
    )
    parser.add_argument("--optflow-highpass", action="store_true", help="Force-enable high-pass preprocessing in optical flow")
    parser.add_argument("--optflow-no-highpass", action="store_true", help="Force-disable high-pass preprocessing in optical flow")

    args = parser.parse_args()

    if args.image_path and not os.path.exists(args.image_path):
        print(f"Warning: Image path {args.image_path} does not exist", file=sys.stderr)

    if args.optflow_highpass and args.optflow_no_highpass:
        print("Error: --optflow-highpass and --optflow-no-highpass are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if args.optflow_highpass:
        optflow_highpass = True
    elif args.optflow_no_highpass:
        optflow_highpass = False
    else:
        optflow_highpass = None

    base_name = args.fasta_file.split(".")[0].split("/")[-1]
    output_dir = os.path.join(args.output_path, base_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        build_trees(
            args.fasta_file,
            output_dir,
            args.seq_type,
            args.image_path,
            args.algorithms,
            args.optflow_mode,
            args.optflow_threshold,
            args.optflow_diagonal_width,
            optflow_highpass,
            args.model_path,
        )
    except Exception as e:
        print(f"Error building trees: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
