#!/usr/bin/env python
"""Sanitize FASTA sequences for analysis."""

import argparse
import sys
from biomodelml.sanitize import convert_and_remove_unrelated_sequences


def main():
    """
    Sanitize FASTA sequences by removing unrelated sequences and converting formats.
    
    This tool processes FASTA files to prepare them for phylogenetic analysis,
    removing sequences that don't match the expected type (nucleotide or protein).
    """
    parser = argparse.ArgumentParser(
        description="Sanitize FASTA sequences for phylogenetic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sequences.fasta N
  %(prog)s proteins.fasta P
        """
    )
    
    parser.add_argument(
        "fasta_file",
        help="Path to the FASTA file to sanitize"
    )
    
    parser.add_argument(
        "seq_type",
        choices=["N", "P"],
        help="Sequence type: N for nucleotides, P for proteins"
    )
    
    args = parser.parse_args()
    
    try:
        convert_and_remove_unrelated_sequences(args.fasta_file, args.seq_type)
        print(f"Successfully sanitized {args.fasta_file}")
    except Exception as e:
        print(f"Error sanitizing sequences: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
