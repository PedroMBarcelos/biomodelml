#!/usr/bin/env python
"""Generate self-comparison matrices as images."""

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from Bio import SeqIO
from biomodelml.matrices import save_image_by_matrices


def process_fasta(fasta_file: str, output_path: str, seq_type: str, max_window: int = 255, generate_variations: bool = False):
    """
    Process FASTA file and generate self-comparison matrix images.
    
    Args:
        fasta_file: Path to sanitized FASTA file
        output_path: Directory to save generated images
        seq_type: Sequence type ('N' or 'P')
        max_window: Maximum window size for matrix generation
        generate_variations: If True, generate all 11 channel variations.
                            If False (default), generate only full/ RGB image (91% storage savings).
    """
    procs = os.cpu_count()

    with open(fasta_file, "r") as handle:
        sequences = list(SeqIO.parse(handle, "fasta"))
        print(f"File with {len(sequences)} sequences")

    with open(fasta_file) as handle:
        sequences = SeqIO.parse(handle, "fasta")
        to_run = []
        for s in sequences:
            to_run.append(
                (s.description, s.description, s.seq, s.seq, max_window, output_path, seq_type, generate_variations)
            )
        
        print(f"Starting to build image matrix for {len(to_run)} sequences")
        
        try:
            with ThreadPoolExecutor(max_workers=procs) as pool:
                futures = [pool.submit(save_image_by_matrices, *data) for data in to_run]
            [f.result() for f in futures]
            print(f"Successfully generated {len(to_run)} matrix images in {output_path}")
        except Exception as e:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            raise e


def main():
    """Generate self-comparison matrix images from FASTA sequences."""
    parser = argparse.ArgumentParser(
        description="Generate self-comparison matrix images from FASTA sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sequences.fasta.N.sanitized output/ N
  %(prog)s proteins.fasta.P.sanitized output/ P --max-window 512
        """
    )
    
    parser.add_argument(
        "fasta_file",
        help="Path to the sanitized FASTA file"
    )
    
    parser.add_argument(
        "output_path",
        help="Directory to save the generated matrix images"
    )
    
    parser.add_argument(
        "seq_type",
        choices=["N", "P"],
        help="Sequence type: N for nucleotides, P for proteins"
    )
    
    parser.add_argument(
        "--max-window",
        type=int,
        default=255,
        help="Maximum window size for matrix generation (default: 255)"
    )
    
    parser.add_argument(
        "--extended-channels",
        action="store_true",
        help="Generate all 11 image channel variations (red/, green/, blue/, gray_*, etc.). "
             "Default: generate only full/ RGB images (saves ~91%% disk space and I/O time)."
    )
    
    args = parser.parse_args()
    
    # Create output subdirectory based on fasta filename
    base_name = args.fasta_file.split(".")[0].split("/")[-1]
    output_dir = os.path.join(args.output_path, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        process_fasta(args.fasta_file, output_dir, args.seq_type, args.max_window, args.extended_channels)
    except Exception as e:
        print(f"Error generating matrices: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
