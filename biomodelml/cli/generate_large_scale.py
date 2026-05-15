#!/usr/bin/env python
"""CLI for large-scale sequence generation."""

import argparse
import sys
import logging
from pathlib import Path

from biomodelml.data_generation.large_scale_generator import LargeScaleSequenceGenerator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Generate 100K+ protein sequences with randomized evolutionary parameters."""
    parser = argparse.ArgumentParser(
        description="Generate 100K+ protein sequences with randomized evolutionary parameters for CNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100K sequences (default)
  biomodelml-generate-large-scale output_dir/
  
  # Generate 50K sequences in 500-sequence batches
  biomodelml-generate-large-scale output_dir/ --target-sequences 50000 --batch-size 500
  
  # Use 8 parallel workers
  biomodelml-generate-large-scale output_dir/ --num-workers 8
  
  # Resume incomplete generation
  biomodelml-generate-large-scale output_dir/ --resume
  
  # Set reproducible seed
  biomodelml-generate-large-scale output_dir/ --seed 42
        """
    )

    parser.add_argument(
        "output_dir",
        help="Root directory for all outputs (sequences, trees, metadata, distances)"
    )

    parser.add_argument(
        "--target-sequences",
        type=int,
        default=100000,
        help="Total number of sequences to generate (default: 100000)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of sequences per batch file (default: 1000)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for tree generation (default: 4)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master random seed for reproducibility (optional)"
    )

    parser.add_argument(
        "--iqtree-path",
        default="iqtree",
        help="Path to IQ-TREE executable (default: assume in PATH)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete generation (checks for checkpoint)"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.target_sequences < 1:
        print("Error: --target-sequences must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.num_workers < 1:
        print("Error: --num-workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Check if output exists and resume logic
    checkpoint_file = output_dir / "metadata" / "checkpoint.json"
    if output_dir.exists() and not args.resume and checkpoint_file.exists():
        print(
            f"Output directory {output_dir} already contains a checkpoint.\n"
            f"Use --resume to continue, or remove the directory and start fresh.",
            file=sys.stderr
        )
        sys.exit(1)

    # Initialize generator
    try:
        generator = LargeScaleSequenceGenerator(
            target_sequences=args.target_sequences,
            batch_size=args.batch_size,
            sequence_type="P",  # Protein only
            num_workers=args.num_workers,
            iqtree_path=args.iqtree_path,
            random_seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate
    try:
        logger.info(f"Starting large-scale sequence generation")
        logger.info(f"  Target: {args.target_sequences:,} sequences")
        logger.info(f"  Batch size: {args.batch_size:,} sequences per batch")
        logger.info(f"  Sequence type: Protein (P)")
        logger.info(f"  Parallel workers: {args.num_workers}")
        if args.seed:
            logger.info(f"  Master seed: {args.seed}")

        gen_log = generator.generate(str(output_dir))

        logger.info("")
        logger.info("=" * 70)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Generated: {gen_log['total_sequences_generated']:,} sequences")
        logger.info(f"Batches: {gen_log['num_batches']}")
        logger.info(f"Elapsed: {gen_log['elapsed_seconds']:.1f} seconds")
        logger.info(f"Output: {output_dir}")
        logger.info("")
        logger.info("Output structure:")
        logger.info(f"  sequences/     - FASTA files (batch_*.fasta)")
        logger.info(f"  trees/         - Newick tree files for each tree")
        logger.info(f"  distances/     - Pairwise distance matrices (CSV)")
        logger.info(f"  metadata/      - JSON logs and checkpoint")

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
