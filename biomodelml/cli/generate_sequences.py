#!/usr/bin/env python
"""CLI command for generating synthetic sequences with AliSim."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from biomodelml.data_generation.presets import (
    get_preset,
    validate_preset_overrides,
    NUCLEOTIDE_MODELS,
    PROTEIN_MODELS,
    TREE_STYLES,
    EVOLUTION_RATES,
)
from biomodelml.data_generation.alisim_generator import AliSimGenerator


def main():
    """Generate synthetic aligned sequences using AliSim."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic aligned sequences for training deep learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 replicates with default training preset
  biomodelml-generate-sequences output/ --num-replicates 10
  
  # Generate benchmark dataset with custom alignment length
  biomodelml-generate-sequences output/ --preset benchmark --alignment-length 2000
  
  # Generate high-noise dataset for protein sequences
  biomodelml-generate-sequences output/ --preset noise --sequence-type P --num-sequences 30
  
  # Generate with custom model and reproducible seed
  biomodelml-generate-sequences output/ --model HKY --seed 42
        """
    )
    
    parser.add_argument(
        "output_dir",
        help="Root directory to store sequences, trees, and metadata"
    )
    
    parser.add_argument(
        "--preset",
        choices=["training", "benchmark", "noise"],
        default="training",
        help="Preset configuration (default: training)"
    )
    
    parser.add_argument(
        "--num-replicates",
        type=int,
        default=1,
        help="Number of independent sequence replicates (default: 1)"
    )
    
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=None,
        help="Number of sequences per alignment (default: from preset)"
    )
    
    parser.add_argument(
        "--alignment-length",
        type=int,
        default=None,
        help="Sequence length in bp/aa (default: from preset)"
    )
    
    parser.add_argument(
        "--evolution-rate",
        choices=EVOLUTION_RATES,
        default=None,
        help="Override evolution rate (default: from preset)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Substitution model (default: from preset)"
    )
    
    parser.add_argument(
        "--tree-style",
        choices=TREE_STYLES,
        default=None,
        help="Tree topology (default: from preset)"
    )
    
    parser.add_argument(
        "--sequence-type",
        choices=["N", "P"],
        default="N",
        help="Sequence type: N=nucleotide, P=protein (default: N)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    parser.add_argument(
        "--iqtree-path",
        default="iqtree2",
        help="Path to iqtree2 executable (default: assume in PATH)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and override preset
    try:
        base_config = get_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = validate_preset_overrides(
        base_config,
        alignment_length=args.alignment_length,
        evolution_rate=args.evolution_rate,
        model=args.model,
        tree_style=args.tree_style,
        num_sequences=args.num_sequences,
    )
    
    # Validate model choice
    if args.sequence_type == "N":
        if config.model not in NUCLEOTIDE_MODELS:
            print(
                f"Warning: Model '{config.model}' may not be suitable for nucleotides. "
                f"Available models: {', '.join(NUCLEOTIDE_MODELS)}",
                file=sys.stderr
            )
    else:
        if config.model not in PROTEIN_MODELS:
            print(
                f"Warning: Model '{config.model}' may not be suitable for proteins. "
                f"Available models: {', '.join(PROTEIN_MODELS)}",
                file=sys.stderr
            )
    
    # Initialize generator
    try:
        generator = AliSimGenerator(iqtree_executable=args.iqtree_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate sequences
    print(f"Generating {args.num_replicates} replicates with {args.preset} preset...")
    print(f"  Model: {config.model}")
    print(f"  Alignment length: {config.alignment_length}")
    print(f"  Tree style: {config.tree_style}")
    print(f"  Evolution rate: {config.evolution_rate}")
    
    # Create output subdirectories
    sequences_dir = output_dir / "sequences"
    trees_dir = output_dir / "trees"
    metadata_dir = output_dir / "metadata"
    
    sequences_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate replicates
    jobs = []
    for rep_idx in range(args.num_replicates):
        rep_num = f"{rep_idx + 1:03d}"
        rep_sequences_dir = sequences_dir / f"replicate_{rep_num}"
        rep_trees_dir = trees_dir / f"replicate_{rep_num}"
        
        rep_sequences_dir.mkdir(parents=True, exist_ok=True)
        rep_trees_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sequences for this replicate
        seed = args.seed + rep_idx if args.seed is not None else None
        job_id = f"{args.preset}_{rep_num}_001"
        
        try:
            # Note: In this implementation, AliSimGenerator.generate() saves to output_dir
            # We modify it to save to the replicate-specific directory
            job = generator.generate(
                output_dir=rep_sequences_dir,
                config=config,
                job_id=f"alignment_001",
                sequence_type=args.sequence_type,
                random_seed=seed,
            )
            
            # Move tree to trees directory
            import shutil
            fasta_src = rep_sequences_dir / f"alignment_001.fasta"
            tree_src = rep_sequences_dir / f"alignment_001.newick"
            dist_src = rep_sequences_dir / f"alignment_001.distances.csv"
            
            tree_dst = rep_trees_dir / f"alignment_001.newick"
            dist_dst = rep_trees_dir / f"alignment_001.distances.csv"
            
            if tree_src.exists():
                shutil.move(str(tree_src), str(tree_dst))
            if dist_src.exists():
                shutil.move(str(dist_src), str(dist_dst))
            
            # Update job with correct paths
            job.fasta_path = str(fasta_src)
            job.tree_path = str(tree_dst)
            job.distances_path = str(dist_dst)
            job.job_id = job_id
            job.sequence_id = job_id
            
            jobs.append(job)
            print(f"✓ Replicate {rep_num}")
            
        except Exception as e:
            print(f"✗ Error generating replicate {rep_num}: {e}", file=sys.stderr)
            continue
    
    # Save generation manifest
    manifest = {
        "generation_date": datetime.now().isoformat(),
        "preset": args.preset,
        "num_replicates": args.num_replicates,
        "num_replicates_generated": len(jobs),
        "configuration": config.to_dict(),
        "sequence_type": args.sequence_type,
        "random_seed": args.seed,
        "jobs": [
            {
                "job_id": job.job_id,
                "fasta_path": job.fasta_path,
                "tree_path": job.tree_path,
                "distances_path": job.distances_path,
                "num_sequences": job.num_sequences,
                "alignment_length": job.alignment_length,
                "random_seed": job.random_seed,
                "timestamp": job.timestamp,
            }
            for job in jobs
        ],
    }
    
    manifest_file = metadata_dir / "generation_log.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Generated {len(jobs)} replicates")
    print(f"✓ Output saved to {output_dir}")
    print(f"✓ Manifest saved to {manifest_file}")
    
    if len(jobs) < args.num_replicates:
        print(f"\nWarning: Only {len(jobs)}/{args.num_replicates} replicates succeeded", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
