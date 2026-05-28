#!/usr/bin/env python
"""CLI command for generating images from FASTA sequences."""

import argparse
import sys
from pathlib import Path

from biomodelml.data_generation.image_generator import ImageGenerator
from biomodelml.sanitize import convert_and_remove_unrelated_sequences


def _fasta_record_count(fasta_file: Path) -> int:
    """Count FASTA records by header lines."""
    count = 0
    with fasta_file.open("r") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _find_sanitized_files(input_path: Path, seq_type: str):
    """Sanitize FASTA inputs and return sanitized files to process."""
    if input_path.is_file():
        if input_path.name.endswith(".sanitized"):
            return [input_path]

        convert_and_remove_unrelated_sequences(str(input_path), seq_type)
        sanitized_file = Path(f"{input_path}.{seq_type}.sanitized")
        if _fasta_record_count(sanitized_file) == 0:
            raise ValueError(
                f"No valid sequences after sanitization for {input_path}. "
                f"Check sequence type '{seq_type}' and FASTA content."
            )
        return [sanitized_file]

    fasta_files = sorted(list(input_path.glob("**/*.fasta")) + list(input_path.glob("**/*.fa")))

    sanitized_files = []
    if fasta_files:
        print(f"Sanitizing {len(fasta_files)} FASTA files...")
        for fasta_file in fasta_files:
            convert_and_remove_unrelated_sequences(str(fasta_file), seq_type)
            sanitized_file = Path(f"{fasta_file}.{seq_type}.sanitized")
            if _fasta_record_count(sanitized_file) > 0:
                sanitized_files.append(sanitized_file)

        if not sanitized_files:
            raise ValueError(
                "No valid sequences after sanitization for any FASTA file. "
                f"Check sequence type '{seq_type}' and input content."
            )
        return sanitized_files

    sanitized_files = sorted(input_path.glob(f"**/*.{seq_type}.sanitized"))
    if sanitized_files:
        sanitized_files = [f for f in sanitized_files if _fasta_record_count(f) > 0]
        if not sanitized_files:
            raise ValueError(
                f"No valid sequences found in existing '.{seq_type}.sanitized' files."
            )
        return sanitized_files

    raise FileNotFoundError(f"No FASTA files found in {input_path}")


def main():
    """Generate RGB matrices from FASTA files."""
    parser = argparse.ArgumentParser(
        description="Generate RGB sequence matrices from FASTA files for training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images from sequences produced by biomodelml-generate-sequences
  biomodelml-generate-images output/sequences/ output/ N
  
  # Generate images from custom FASTA file
  biomodelml-generate-images mysequences.fasta.N.sanitized output/ N
  
  # Use larger matrix size for high-resolution images
  biomodelml-generate-images output/sequences/ output/ N --max-window 512
  
  # Use more parallel workers for faster processing
  biomodelml-generate-images output/sequences/ output/ N --num-workers 8
        """
    )
    
    parser.add_argument(
        "fasta_dir_or_file",
        help="Input FASTA directory or file"
    )
    
    parser.add_argument(
        "output_dir",
        help="Output directory (will contain images/ and metadata/ subdirectories)"
    )
    
    parser.add_argument(
        "seq_type",
        choices=["N", "P"],
        help="Sequence type: N=nucleotide, P=protein"
    )
    
    parser.add_argument(
        "--max-window",
        type=int,
        default=255,
        help="Maximum matrix dimension (default: 255)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Link to tree distances if available (default: True)"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.fasta_dir_or_file)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # If single FASTA file, use parent directory
    if input_path.is_file():
        input_dir = input_path.parent
        print(f"Processing FASTA file: {input_path}")
    else:
        input_dir = input_path
        print(f"Processing FASTA directory: {input_dir}")

    # Ensure all inputs are sanitized before matrix generation
    try:
        sanitized_files = _find_sanitized_files(input_path, args.seq_type)
    except Exception as e:
        print(f"Error sanitizing input sequences: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize generator
    try:
        generator = ImageGenerator(
            output_dir=args.output_dir,
            max_window=args.max_window,
            num_workers=args.num_workers,
            include_metadata=args.include_metadata,
        )
    except OSError as e:
        print(f"Error: Could not create output directory: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate images
    print(f"Generating images with max_window={args.max_window}...")
    
    try:
        generator.generate_from_files(
            fasta_files=sanitized_files,
            sequence_type=args.seq_type,
            link_tree_distances=args.include_metadata,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during image generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✓ Image generation complete")
    print(f"✓ Output saved to {args.output_dir}")
    
    # Show manifest location
    manifest_path = Path(args.output_dir) / "metadata" / "image_manifest.json"
    if manifest_path.exists():
        print(f"✓ Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
