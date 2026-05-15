"""Large-scale synthetic sequence generation for deep learning training.

Generates 100K+ protein sequences across randomized evolutionary parameters
with full parallelization, metadata tracking, and resume capability.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from joblib import Parallel, delayed

from biomodelml.data_generation.alisim_generator import AliSimGenerator
from biomodelml.data_generation.presets import AliSimConfig


logger = logging.getLogger(__name__)

# Use only protein models confirmed to work with IQ-TREE alisim
WORKING_PROTEIN_MODELS = ["LG"]


@dataclass
class GenerationParameters:
    """Random parameters for a single tree's sequence generation."""
    model: str
    tree_style: str
    evolution_rate: str
    alignment_length: int
    num_sequences: int
    random_seed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LargeScaleSequenceGenerator:
    """Generates 100K+ protein sequences with randomized evolutionary parameters."""

    def __init__(
        self,
        target_sequences: int = 100000,
        batch_size: int = 1000,
        sequence_type: str = "P",
        num_workers: int = 4,
        iqtree_path: str = "iqtree",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize large-scale generator.

        Args:
            target_sequences: Total number of sequences to generate
            batch_size: Sequences per batch (per FASTA file)
            sequence_type: 'P' for protein only
            num_workers: Number of parallel workers for batch generation
            iqtree_path: Path to IQ-TREE executable
            random_seed: Master seed for reproducibility
        """
        if sequence_type != "P":
            raise ValueError("Only protein sequences ('P') are supported currently")

        self.target_sequences = target_sequences
        self.batch_size = batch_size
        self.sequence_type = sequence_type
        self.num_workers = num_workers
        self.iqtree_path = iqtree_path
        self.random_seed = random_seed if random_seed is not None else np.random.randint(1, 2**31 - 1)

        # Set numpy random seed for reproducibility
        np.random.seed(self.random_seed)

        # Initialize AliSimGenerator
        try:
            self.alisim_gen = AliSimGenerator(iqtree_executable=iqtree_path)
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize AliSimGenerator: {e}")
            raise

    def generate(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate 100K+ sequences in batches with randomized parameters.

        Args:
            output_dir: Root directory for all outputs

        Returns:
            Dictionary with generation statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup output subdirectories
        sequences_dir = output_dir / "sequences"
        trees_dir = output_dir / "trees"
        distances_dir = output_dir / "distances"
        metadata_dir = output_dir / "metadata"

        sequences_dir.mkdir(exist_ok=True)
        trees_dir.mkdir(exist_ok=True)
        distances_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)

        # Load checkpoint if exists
        checkpoint_file = metadata_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            batches_completed = checkpoint.get("batches_completed", 0)
            total_sequences_generated = checkpoint.get("total_sequences", 0)
            logger.info(f"Resuming from checkpoint: {batches_completed} batches, {total_sequences_generated} sequences")
        else:
            batches_completed = 0
            total_sequences_generated = 0
            checkpoint = {}

        # Calculate batches needed
        num_batches = (self.target_sequences + self.batch_size - 1) // self.batch_size

        # Initialize metadata tracking
        batch_metadata = {}
        parameters_used = {}
        start_time = datetime.now()

        # Generate batches
        logger.info(f"Starting generation: target {self.target_sequences} sequences in {num_batches} batches")

        for batch_idx in range(batches_completed, num_batches):
            batch_num = batch_idx + 1
            logger.info(f"Generating batch {batch_num}/{num_batches}")

            # Determine number of sequences for this batch
            remaining = self.target_sequences - total_sequences_generated
            batch_target = min(self.batch_size, remaining)

            # Generate this batch
            batch_info = self._generate_batch(
                batch_idx=batch_idx,
                target_sequences=batch_target,
                output_dir=output_dir,
                sequences_dir=sequences_dir,
                trees_dir=trees_dir,
                distances_dir=distances_dir,
            )

            batch_metadata[f"batch_{batch_num:03d}"] = batch_info
            total_sequences_generated += batch_info["sequences_generated"]

            # Track parameter usage
            for param_key, param_val in batch_info["parameters"]:
                if param_key not in parameters_used:
                    parameters_used[param_key] = {"count": 0, "details": param_val}
                parameters_used[param_key]["count"] += 1

            # Save checkpoint
            checkpoint = {
                "batches_completed": batch_num,
                "total_sequences": total_sequences_generated,
                "timestamp": datetime.now().isoformat(),
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(f"  Batch {batch_num}: {batch_info['sequences_generated']} sequences generated")

        # Save final metadata
        elapsed = datetime.now() - start_time

        generation_log = {
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "target_sequences": self.target_sequences,
            "total_sequences_generated": total_sequences_generated,
            "num_batches": num_batches,
            "batch_size": self.batch_size,
            "sequence_type": self.sequence_type,
            "master_seed": self.random_seed,
            "num_workers": self.num_workers,
            "iqtree_path": self.iqtree_path,
        }

        with open(metadata_dir / "generation_log.json", "w") as f:
            json.dump(generation_log, f, indent=2)

        with open(metadata_dir / "batch_metadata.json", "w") as f:
            json.dump(batch_metadata, f, indent=2)

        with open(metadata_dir / "parameters_used.json", "w") as f:
            json.dump(parameters_used, f, indent=2)

        logger.info(f"✓ Generation complete: {total_sequences_generated} sequences in {elapsed.total_seconds():.1f}s")
        logger.info(f"✓ Output saved to {output_dir}")

        return generation_log

    def _generate_batch(
        self,
        batch_idx: int,
        target_sequences: int,
        output_dir: Path,
        sequences_dir: Path,
        trees_dir: Path,
        distances_dir: Path,
    ) -> Dict[str, Any]:
        """
        Generate a single batch of sequences with random parameters.

        Returns:
            Dictionary with batch metadata
        """
        batch_num = batch_idx + 1

        # Estimate how many trees to generate for this batch
        # Average sequences per tree: random between 5-50
        avg_seqs_per_tree = np.mean([5, 50])
        num_trees = max(1, int(target_sequences / avg_seqs_per_tree))

        # Generate random parameters for each tree
        tree_params = [
            (i, self._randomize_parameters(batch_idx, i))
            for i in range(num_trees)
        ]

        # Create batch-specific directories
        batch_trees_dir = trees_dir / f"batch_{batch_num:03d}_trees"
        batch_distances_dir = distances_dir / f"batch_{batch_num:03d}"
        batch_trees_dir.mkdir(exist_ok=True)
        batch_distances_dir.mkdir(exist_ok=True)

        # Generate trees in parallel
        logger.info(f"  Generating {num_trees} trees in parallel (max {self.num_workers} workers)")
        results = Parallel(n_jobs=self.num_workers)(
            delayed(self._generate_tree)(
                tree_idx, params, batch_num, batch_trees_dir, batch_distances_dir
            )
            for tree_idx, params in tree_params
        )

        # Aggregate results
        all_fastas = []
        total_seqs = 0
        param_list = []

        for tree_result in results:
            if tree_result is not None:
                all_fastas.append(tree_result["fasta_path"])
                total_seqs += tree_result["num_sequences"]
                param_list.append((
                    f"batch_{batch_num:03d}_tree_{tree_result['tree_idx']:03d}",
                    tree_result["parameters"]
                ))

        # Merge FASTA files into batch file
        batch_fasta = sequences_dir / f"batch_{batch_num:03d}.fasta"
        self._merge_fastas(all_fastas, batch_fasta)

        return {
            "batch_number": batch_num,
            "sequences_generated": total_seqs,
            "num_trees": len(results),
            "trees_dir": str(batch_trees_dir),
            "distances_dir": str(batch_distances_dir),
            "fasta_path": str(batch_fasta),
            "parameters": param_list,
        }

    def _generate_tree(
        self,
        tree_idx: int,
        params: GenerationParameters,
        batch_num: int,
        batch_trees_dir: Path,
        batch_distances_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Generate sequences for a single tree with given parameters."""
        try:
            tree_num_str = f"{tree_idx + 1:03d}"

            # Create config from parameters
            config = AliSimConfig(
                alignment_length=params.alignment_length,
                evolution_rate=params.evolution_rate,
                model=params.model,
                tree_style=params.tree_style,
                num_sequences=params.num_sequences,
                description=f"Large-scale batch {batch_num} tree {tree_num_str}",
            )

            # Generate sequences
            job_id = f"batch_{batch_num:03d}_tree_{tree_num_str}"
            job = self.alisim_gen.generate(
                output_dir=str(batch_trees_dir),
                config=config,
                job_id=job_id,
                sequence_type=self.sequence_type,
                random_seed=params.random_seed,
            )

            # Copy distances CSV to batch distances directory
            import shutil
            distances_src = Path(job.distances_path)
            distances_dst = batch_distances_dir / f"tree_{tree_num_str}_distances.csv"
            if distances_src.exists():
                shutil.copy(distances_src, distances_dst)

            return {
                "tree_idx": tree_idx,
                "num_sequences": job.num_sequences,
                "fasta_path": job.fasta_path,
                "tree_path": job.tree_path,
                "distances_path": str(distances_dst),
                "parameters": params.to_dict(),
            }

        except Exception as e:
            logger.warning(f"Failed to generate tree {tree_idx}: {e}")
            return None

    @staticmethod
    def _merge_fastas(fasta_files: List[str], output_fasta: Path) -> None:
        """Merge multiple FASTA files into one."""
        with open(output_fasta, "w") as outf:
            for fasta_file in fasta_files:
                if Path(fasta_file).exists():
                    with open(fasta_file) as inf:
                        outf.write(inf.read())

    def _randomize_parameters(self, batch_idx: int, tree_idx: int) -> GenerationParameters:
        """Generate random parameters for a tree, seeded by batch and tree index."""
        # Use deterministic seed based on batch and tree for reproducibility
        seed = int(
            np.abs(
                np.sin(batch_idx * 1000 + tree_idx) * 2**31
            )
        ) % (2**31 - 1)
        local_rng = np.random.RandomState(seed)

        model = local_rng.choice(WORKING_PROTEIN_MODELS)
        tree_style = local_rng.choice(["balanced", "random", "power-law"])
        evolution_rate = local_rng.choice(["slow", "moderate", "fast", "very_fast"])

        # Log-uniform distribution for alignment length (200-2000)
        alignment_length = int(
            np.exp(local_rng.uniform(np.log(200), np.log(2000)))
        )

        # Log-uniform distribution for num_sequences (5-50)
        num_sequences = int(
            np.exp(local_rng.uniform(np.log(5), np.log(50)))
        )

        # Ensure num_sequences is at least 2
        num_sequences = max(2, num_sequences)

        random_seed = int(local_rng.randint(1, 2**31 - 1))

        return GenerationParameters(
            model=model,
            tree_style=tree_style,
            evolution_rate=evolution_rate,
            alignment_length=alignment_length,
            num_sequences=num_sequences,
            random_seed=random_seed,
        )
