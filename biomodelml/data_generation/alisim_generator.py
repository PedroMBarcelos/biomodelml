"""
AliSim wrapper for synthetic sequence generation via IQ-TREE.

This module provides high-level interfaces for generating aligned sequences
with known evolutionary distances using IQ-TREE's AliSim simulator.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from Bio import SeqIO
import shutil

from biomodelml.data_generation.presets import AliSimConfig
from biomodelml.structs import SequenceGenerationJob


class AliSimGenerator:
    """Wrapper around IQ-TREE's AliSim sequence simulator.
    
    Handles subprocess calls to iqtree3 with AliSim, parses output trees
    and FASTA files, and computes pairwise distances from alignments.
    """
    
    def __init__(self, iqtree_executable: str = "iqtree3"):
        """
        Initialize AliSim generator.
        
        Args:
            iqtree_executable: Path to iqtree3 binary (default: assumes it's in PATH)
            
        Raises:
            FileNotFoundError: If iqtree3 cannot be found
        """
        self.iqtree_path = iqtree_executable
        self._check_iqtree_available()
    
    def _check_iqtree_available(self) -> None:
        """
        Check if an IQ-TREE binary is available with AliSim support.
        Prioritizes iqtree2 (v3.1.1) and iqtree3 which have --alisim support
        over iqtree 1.6.12 which does not.

        Raises:
            FileNotFoundError: If no working IQ-TREE binary with AliSim can be found
        """
        tried = []

        # Helper to test if a candidate supports --alisim
        def _test_has_alisim(candidate: str) -> bool:
            try:
                result = subprocess.run(
                    [candidate, "--help"],
                    capture_output=True,
                    timeout=5,
                    text=True,
                )
                return "--alisim" in result.stdout or "--alisim" in result.stderr
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False

        # First try ordered list: iqtree2, iqtree3, then iqtree (fallback)
        # Always try this order regardless of what was passed as iqtree_executable
        candidates = ["iqtree2", "iqtree3", "iqtree"]
        
        for name in candidates:
            found = shutil.which(name)
            if found and _test_has_alisim(found):
                self.iqtree_path = found
                return
            
            # If which didn't return a path, still try the bare name
            if _test_has_alisim(name):
                self.iqtree_path = name
                return

            tried.append(name)

        # Nothing worked
        raise FileNotFoundError(
            f"Could not find a working IQ-TREE binary. Tried: {', '.join(tried)}. "
            f"Install IQ-TREE from: http://www.iqtree.org/ or specify custom path: "
            f"AliSimGenerator(iqtree_executable='/path/to/iqtree')"
        )
    
    def generate(
        self,
        output_dir: str,
        config: AliSimConfig,
        job_id: str,
        sequence_type: str = "N",
        random_seed: Optional[int] = None,
    ) -> SequenceGenerationJob:
        """
        Generate synthetic aligned sequences using AliSim.
        
        Args:
            output_dir: Directory to save FASTA and tree files
            config: AliSimConfig with generation parameters
            job_id: Unique identifier for this generation run
            sequence_type: 'N' for nucleotide, 'P' for protein
            random_seed: Random seed for reproducibility (optional)
            
        Returns:
            SequenceGenerationJob with paths to outputs
            
        Raises:
            subprocess.CalledProcessError: If alisim fails
            ValueError: If sequence_type is invalid
            OSError: If output directory cannot be created
        """
        if sequence_type not in ("N", "P"):
            raise ValueError(f"sequence_type must be 'N' or 'P', got {sequence_type}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary working directory for alisim
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run AliSim
            tree_file, aln_file = self._run_alisim(
                output_dir=tmpdir,
                config=config,
                sequence_type=sequence_type,
                random_seed=random_seed,
            )
            
            # Copy outputs to final location
            final_fasta = output_dir / f"{job_id}.fasta"
            final_tree = output_dir / f"{job_id}.newick"
            
            # Copy files
            import shutil
            shutil.copy(aln_file, final_fasta)
            shutil.copy(tree_file, final_tree)
            
            # Compute pairwise distances
            distances_csv = output_dir / f"{job_id}.distances.csv"
            self._compute_distances(final_fasta, distances_csv, sequence_type)
            
            # Create job record
            job = SequenceGenerationJob(
                job_id=job_id,
                preset=config.description,
                sequence_id=job_id,
                fasta_path=str(final_fasta),
                tree_path=str(final_tree),
                distances_path=str(distances_csv),
                num_sequences=config.num_sequences,
                alignment_length=config.alignment_length,
                sequence_type=sequence_type,
                random_seed=random_seed,
                parameters=config.to_dict(),
                timestamp=datetime.now().isoformat(),
            )
            
            return job
    
    def _run_alisim(
        self,
        output_dir: Path,
        config: AliSimConfig,
        sequence_type: str,
        random_seed: Optional[int] = None,
    ) -> Tuple[Path, Path]:
        """
        Execute alisim command to generate sequences.
        
        Args:
            output_dir: Temporary output directory
            config: Generation configuration
            sequence_type: 'N' or 'P'
            random_seed: Random seed (optional)
            
        Returns:
            Tuple of (tree_file_path, alignment_file_path)
            
        Raises:
            subprocess.CalledProcessError: If alisim fails
        """
        # Construct alisim command
        # Generate tree string and write to a temporary tree file
        tree_str = self._generate_tree_string(config, sequence_type, random_seed)
        tree_input = output_dir / "input.tree"
        with open(tree_input, 'w') as tf:
            tf.write(tree_str + "\n")

        cmd = [
            self.iqtree_path,
            "--alisim", str(output_dir / "output"),
            "-m", config.model,
            "-t", str(tree_input),
            "--length", str(config.alignment_length),
            "-af", "fasta",
        ]
        
        # Add sequence type specific arguments
        if sequence_type == "N":
            # Nucleotide - no additional flags needed
            pass
        elif sequence_type == "P":
            # Protein - output amino acids
            cmd.append("--seqtype")
            cmd.append("AA")
        
        # Run alisim
        # Ensure a seed is provided (IQ-TREE may require it for non-zero exit)
        seed = random_seed if random_seed is not None else int(np.random.randint(1, 2**31 - 1))
        cmd.extend(["-seed", str(seed)])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                # Use positional arguments for broad compatibility across
                # Python versions/implementations where keyword args
                # like `stdout`/`stderr` may not be accepted.
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    result.stdout,
                    result.stderr,
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"AliSim timed out after 300 seconds")
        
        # Find output alignment file (fasta preferred, fallback to phylip)
        possible_alignments = [output_dir / "output.fasta", output_dir / "output.fa", output_dir / "output.phy"]
        alignment_file = None
        for p in possible_alignments:
            if p.exists():
                alignment_file = p
                break

        if alignment_file is None:
            raise FileNotFoundError(
                f"AliSim did not produce an alignment file. Stdout: {result.stdout}\nStderr: {result.stderr}"
            )

        # Ensure a tree file is available: prefer any treefile produced, else copy the input tree
        tree_file = output_dir / "output.treefile"
        if not tree_file.exists():
            tree_file = output_dir / "output.newick"
            # Write the tree we passed in as the output tree
            with open(tree_file, 'w') as tf:
                tf.write(tree_str + "\n")
        
        return tree_file, alignment_file
    
    def _generate_tree_string(
        self,
        config: AliSimConfig,
        sequence_type: str,
        random_seed: Optional[int],
    ) -> str:
        """
        Generate Newick tree string based on config.
        
        For now, returns a balanced tree string that AliSim can parse.
        In future, could integrate with ETE3 for more complex tree generation.
        
        Args:
            config: Generation configuration
            sequence_type: 'N' or 'P'
            random_seed: Random seed
            
        Returns:
            Newick format tree string
        """
        # Create a simple balanced tree with config.num_sequences leaves
        # AliSim will use this as the evolutionary template
        n_seqs = config.num_sequences
        
        # Simple binary tree generation
        if config.tree_style == "balanced":
            return self._balanced_tree(n_seqs)
        elif config.tree_style == "random":
            return self._random_tree(n_seqs, seed=random_seed)
        elif config.tree_style == "power-law":
            return self._power_law_tree(n_seqs, seed=random_seed)
        else:
            return self._balanced_tree(n_seqs)
    
    @staticmethod
    def _balanced_tree(num_seqs: int) -> str:
        """Generate a balanced binary tree."""
        if num_seqs == 1:
            return "(seq1:0.1);"
        # Create leaves with branch lengths
        leaves = [f"seq{i+1}:0.1" for i in range(num_seqs)]

        # Pairwise combine to build a balanced topology without redundant
        # internal branch lengths that some IQ-TREE versions reject.
        nodes = leaves[:]
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    new_nodes.append(f"({nodes[i]},{nodes[i+1]})")
                else:
                    new_nodes.append(nodes[i])
            nodes = new_nodes

        return f"{nodes[0]};"
    
    @staticmethod
    def _random_tree(num_seqs: int, seed: Optional[int] = None) -> str:
        """Generate a random topology tree."""
        if seed is not None:
            np.random.seed(seed)
        
        leaves = [f"seq{i+1}:0.1" for i in range(num_seqs)]
        np.random.shuffle(leaves)
        
        tree_str = leaves[0]
        for leaf in leaves[1:]:
            tree_str = f"({tree_str},{leaf}):0.1"
        
        return f"({tree_str});"
    
    @staticmethod
    def _power_law_tree(num_seqs: int, seed: Optional[int] = None) -> str:
        """Generate a power-law (non-balanced) tree."""
        if seed is not None:
            np.random.seed(seed)
        
        # Power-law style: one long branch with others hanging off
        leaves = [f"seq{i+1}:0.1" for i in range(num_seqs)]
        
        tree_str = leaves[0]
        for leaf in leaves[1:]:
            # Random branch lengths following power law
            branch_len = 0.1 * (np.random.pareto(1) + 1)
            tree_str = f"({tree_str},{leaf}:{branch_len:.3f}):0.1"
        
        return f"({tree_str});"
    
    @staticmethod
    def _compute_distances(
        fasta_file: Path,
        output_csv: Path,
        sequence_type: str,
    ) -> None:
        """
        Compute pairwise distances from aligned FASTA file.
        
        Uses simple Hamming distance normalized by sequence length.
        
        Args:
            fasta_file: Path to FASTA alignment file
            output_csv: Path to output CSV file
            sequence_type: 'N' or 'P'
        """
        # Read sequences
        seqs = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            seqs[record.id] = str(record.seq)
        
        seq_names = list(seqs.keys())
        n_seqs = len(seq_names)
        
        # Compute pairwise distances
        distances = np.zeros((n_seqs, n_seqs), dtype=np.float64)
        
        for i in range(n_seqs):
            for j in range(i, n_seqs):
                seq1 = seqs[seq_names[i]]
                seq2 = seqs[seq_names[j]]
                
                # Hamming distance normalized by length
                dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Write CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv, 'w') as f:
            # Header
            f.write(','.join([''] + seq_names) + '\n')
            # Rows
            for i, name in enumerate(seq_names):
                f.write(name + ',' + ','.join(f"{distances[i, j]:.6f}" for j in range(n_seqs)) + '\n')
