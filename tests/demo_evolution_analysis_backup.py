#!/usr/bin/env python3
"""
Evolutionary Sequence Demonstration

This script simulates protein sequence evolution along a known phylogenetic tree,
then uses biomodelml to reconstruct the phylogeny and validate the results.

Features:
- Point mutations with configurable rates
- Insertions and deletions (indels)
- Simple selection pressure based on amino acid properties
- Full mutation history tracking
- Comparison of reconstructed vs true phylogenies
- All 7 biomodelml algorithms tested
- Automatic Robinson-Foulds comparison
- Customizable taxa count
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter crashes

import argparse
import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Amino acid properties for selection (simplified)
HYDROPHOBIC = set("AVILMFYWP")
CHARGED = set("DEKR")
POLAR = set("STNQC")
AROMATIC = set("FYW")


@dataclass
class MutationEvent:
    """Record of a single mutation event"""
    position: int
    event_type: str  # 'substitution', 'insertion', 'deletion'
    old_value: str
    new_value: str
    branch: str
    
    def __str__(self):
        if self.event_type == 'substitution':
            return f"{self.old_value}{self.position+1}{self.new_value}"
        elif self.event_type == 'insertion':
            return f"ins{self.position+1}:{self.new_value}"
        else:  # deletion
            return f"del{self.position+1}:{self.old_value}"


@dataclass
class EvolutionaryNode:
    """Represents a node in the phylogenetic tree"""
    name: str
    sequence: str = ""
    parent: Optional['EvolutionaryNode'] = None
    children: List['EvolutionaryNode'] = field(default_factory=list)
    branch_length: float = 0.0
    mutations: List[MutationEvent] = field(default_factory=list)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_all_leaves(self) -> List['EvolutionaryNode']:
        """Get all leaf nodes descending from this node"""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves


class SequenceEvolver:
    """Handles sequence evolution with mutations, indels, and selection"""
    
    def __init__(self, mutation_rate: float = 0.002, indel_rate: float = 0.0005, 
                 selection_strength: float = 0.3, seed: Optional[int] = None):
        """
        Initialize the evolver
        
        Args:
            mutation_rate: Probability of substitution per site per branch unit
            indel_rate: Probability of insertion/deletion per site per branch unit
            selection_strength: Strength of selection against unfavorable changes (0-1)
            seed: Random seed for reproducibility
        """
        self.mutation_rate = mutation_rate
        self.indel_rate = indel_rate
        self.selection_strength = selection_strength
        
        if seed is not None:
            random.seed(seed)
    
    def calculate_fitness_penalty(self, old_aa: str, new_aa: str) -> float:
        """
        Calculate fitness penalty for an amino acid substitution
        
        Higher penalty for drastic changes (e.g., hydrophobic to charged)
        Lower penalty for conservative changes
        """
        # No penalty for same amino acid
        if old_aa == new_aa:
            return 0.0
        
        # High penalty for hydrophobic <-> charged
        if (old_aa in HYDROPHOBIC and new_aa in CHARGED) or \
           (old_aa in CHARGED and new_aa in HYDROPHOBIC):
            return 1.0
        
        # Medium penalty for other drastic changes
        if (old_aa in POLAR and new_aa in HYDROPHOBIC) or \
           (old_aa in HYDROPHOBIC and new_aa in POLAR):
            return 0.6
        
        # Low penalty for conservative changes
        return 0.2
    
    def should_accept_mutation(self, old_aa: str, new_aa: str) -> bool:
        """Decide whether to accept a mutation based on fitness penalty"""
        penalty = self.calculate_fitness_penalty(old_aa, new_aa)
        acceptance_prob = 1.0 - (penalty * self.selection_strength)
        return random.random() < acceptance_prob
    
    def apply_point_mutations(self, sequence: str, branch_length: float, 
                             branch_name: str) -> Tuple[str, List[MutationEvent]]:
        """Apply point mutations to a sequence"""
        seq_list = list(sequence)
        mutations = []
        
        for i in range(len(seq_list)):
            if random.random() < self.mutation_rate * branch_length:
                old_aa = seq_list[i]
                new_aa = random.choice(AMINO_ACIDS)
                
                # Apply selection pressure
                if self.should_accept_mutation(old_aa, new_aa):
                    seq_list[i] = new_aa
                    mutations.append(MutationEvent(
                        position=i,
                        event_type='substitution',
                        old_value=old_aa,
                        new_value=new_aa,
                        branch=branch_name
                    ))
        
        return ''.join(seq_list), mutations
    
    def apply_indels(self, sequence: str, branch_length: float, 
                    branch_name: str) -> Tuple[str, List[MutationEvent]]:
        """Apply insertions and deletions"""
        seq_list = list(sequence)
        mutations = []
        i = 0
        
        while i < len(seq_list):
            # Check for deletion
            if random.random() < self.indel_rate * branch_length:
                del_length = random.randint(1, min(5, len(seq_list) - i))
                deleted = ''.join(seq_list[i:i+del_length])
                mutations.append(MutationEvent(
                    position=i,
                    event_type='deletion',
                    old_value=deleted,
                    new_value='',
                    branch=branch_name
                ))
                del seq_list[i:i+del_length]
                continue
            
            # Check for insertion
            if random.random() < self.indel_rate * branch_length / 2:
                ins_length = random.randint(1, 5)
                inserted = ''.join(random.choice(AMINO_ACIDS) for _ in range(ins_length))
                mutations.append(MutationEvent(
                    position=i,
                    event_type='insertion',
                    old_value='',
                    new_value=inserted,
                    branch=branch_name
                ))
                seq_list.insert(i, inserted)
            
            i += 1
        
        return ''.join(seq_list), mutations
    
    def evolve_sequence(self, sequence: str, branch_length: float, 
                       branch_name: str) -> Tuple[str, List[MutationEvent]]:
        """
        Evolve a sequence along a branch
        
        Returns:
            Tuple of (evolved_sequence, list_of_mutations)
        """
        # Apply point mutations first
        evolved_seq, point_muts = self.apply_point_mutations(sequence, branch_length, branch_name)
        
        # Then apply indels
        evolved_seq, indel_muts = self.apply_indels(evolved_seq, branch_length, branch_name)
        
        return evolved_seq, point_muts + indel_muts


def generate_ancestral_sequence(length: int = 250, seed: Optional[int] = None) -> str:
    """Generate a random ancestral protein sequence"""
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice(AMINO_ACIDS) for _ in range(length))


def build_example_tree() -> EvolutionaryNode:
    r"""
    Build an example phylogenetic tree with 7 taxa
    
    Tree structure:
                    Root
                   /    \
                  /      \
               Node_A   Node_B
               /  \      /  \
              /    \    /    \
           Seq1  Node_C Seq5  Node_D
                  / \         / \
                 /   \       /   \
              Seq2  Seq3  Seq6  Node_E
                              /  \
                           Seq7  Seq8
    """
    root = EvolutionaryNode(name="Root")
    
    # Left subtree
    node_a = EvolutionaryNode(name="Node_A", parent=root, branch_length=0.5)
    seq1 = EvolutionaryNode(name="Seq1", parent=node_a, branch_length=1.2)
    node_c = EvolutionaryNode(name="Node_C", parent=node_a, branch_length=0.8)
    seq2 = EvolutionaryNode(name="Seq2", parent=node_c, branch_length=0.9)
    seq3 = EvolutionaryNode(name="Seq3", parent=node_c, branch_length=1.1)
    
    # Right subtree
    node_b = EvolutionaryNode(name="Node_B", parent=root, branch_length=0.6)
    seq5 = EvolutionaryNode(name="Seq5", parent=node_b, branch_length=1.5)
    node_d = EvolutionaryNode(name="Node_D", parent=node_b, branch_length=0.7)
    seq6 = EvolutionaryNode(name="Seq6", parent=node_d, branch_length=1.0)
    node_e = EvolutionaryNode(name="Node_E", parent=node_d, branch_length=0.5)
    seq7 = EvolutionaryNode(name="Seq7", parent=node_e, branch_length=0.8)
    seq8 = EvolutionaryNode(name="Seq8", parent=node_e, branch_length=0.9)
    
    # Build tree structure
    root.children = [node_a, node_b]
    node_a.children = [seq1, node_c]
    node_c.children = [seq2, seq3]
    node_b.children = [seq5, node_d]
    node_d.children = [seq6, node_e]
    node_e.children = [seq7, seq8]
    
    return root


def build_tree_with_n_taxa(num_taxa: int = 7, seed: Optional[int] = None) -> EvolutionaryNode:
    """
    Build a balanced binary phylogenetic tree with n taxa
    
    Args:
        num_taxa: Number of leaf nodes (taxa) to create
        seed: Random seed for branch length generation
        
    Returns:
        Root node of the phylogenetic tree
    """
    if seed is not None:
        random.seed(seed + 1000)  # Use different seed from sequence evolution
    
    if num_taxa < 2:
        raise ValueError("Need at least 2 taxa")
    
    # Create leaf nodes
    leaves = [EvolutionaryNode(f"Taxon_{i+1}") for i in range(num_taxa)]
    nodes = leaves[:]
    
    # Build tree bottom-up by pairing nodes
    node_counter = 0
    while len(nodes) > 1:
        # Take pairs of nodes and create parent nodes
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                # Create parent for pair
                parent = EvolutionaryNode(f"inner_{node_counter}")
                node_counter += 1
                child1, child2 = nodes[i], nodes[i+1]
                
                # Assign random branch lengths (between 0.1 and 0.5)
                child1.branch_length = random.uniform(0.1, 0.5)
                child2.branch_length = random.uniform(0.1, 0.5)
                child1.parent = parent
                child2.parent = parent
                
                parent.children = [child1, child2]
                new_nodes.append(parent)
            else:
                # Odd one out, keep for next level
                new_nodes.append(nodes[i])
        
        nodes = new_nodes
    
    root = nodes[0]
    root.name = "Root"
    root.branch_length = 0.0
    
    return root


def evolve_along_tree(node: EvolutionaryNode, evolver: SequenceEvolver) -> None:
    """
    Recursively evolve sequences down the tree
    
    The root node should have its sequence set before calling this function
    """
    for child in node.children:
        # Evolve from parent to child
        child.sequence, child.mutations = evolver.evolve_sequence(
            node.sequence,
            child.branch_length,
            f"{node.name}->{child.name}"
        )
        
        # Recursively evolve children
        if not child.is_leaf():
            evolve_along_tree(child, evolver)


def tree_to_newick(node: EvolutionaryNode, include_inner_labels: bool = True) -> str:
    """Convert tree to Newick format"""
    if node.is_leaf():
        return f"{node.name}:{node.branch_length:.4f}"
    
    children_newick = ','.join(tree_to_newick(child, include_inner_labels) 
                               for child in node.children)
    
    if node.parent is None:  # Root
        return f"({children_newick});"
    
    name_part = node.name if include_inner_labels else ""
    return f"({children_newick}){name_part}:{node.branch_length:.4f}"


def save_sequences_fasta(nodes: List[EvolutionaryNode], output_path: Path) -> None:
    """Save leaf sequences to FASTA file"""
    with open(output_path, 'w') as f:
        for node in nodes:
            if node.is_leaf():
                f.write(f">{node.name}\n")
                # Write sequence in 60-character lines
                seq = node.sequence
                for i in range(0, len(seq), 60):
                    f.write(f"{seq[i:i+60]}\n")


def save_mutation_history(root: EvolutionaryNode, output_path: Path) -> None:
    """Save complete mutation history as JSON"""
    def node_to_dict(node: EvolutionaryNode) -> dict:
        return {
            'name': node.name,
            'branch_length': node.branch_length,
            'sequence_length': len(node.sequence),
            'mutations': [
                {
                    'position': m.position,
                    'type': m.event_type,
                    'old': m.old_value,
                    'new': m.new_value,
                    'branch': m.branch
                }
                for m in node.mutations
            ],
            'children': [node_to_dict(child) for child in node.children]
        }
    
    history = node_to_dict(root)
    
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)


def run_biomodelml_analysis(fasta_path: Path, output_dir: Path, 
                            sequence_type: str = "P") -> None:
    """Run biomodelml analysis on the evolved sequences"""
    print(f"\n{'='*60}")
    print("Running biomodelml analysis...")
    print(f"{'='*60}")
    
    # Import biomodelml modules
    try:
        from biomodelml.experiment import Experiment
        from biomodelml.variants.nw import NeedlemanWunschVariant
        from biomodelml.variants.sw import SmithWatermanVariant
        from biomodelml.variants.resized_ssim_multiscale import ResizedSSIMMultiScaleVariant
        from biomodelml.variants.unrestricted_ssim import UnrestrictedSSIMVariant
        from biomodelml.variants.greedy_ssim import GreedySSIMVariant
        from biomodelml.variants.windowed_ssim_multiscale import WindowedSSIMMultiScaleVariant
        from biomodelml.variants.deep_search.variant import DeepSearchVariant
        from biomodelml.sanitize import convert_and_remove_unrelated_sequences
        from biomodelml.matrices import save_image_by_matrices
        from Bio import SeqIO
        from concurrent.futures import ThreadPoolExecutor
        import os
    except ImportError as e:
        print(f"Error importing biomodelml: {e}")
        print("Make sure biomodelml is installed: pip install -e .")
        sys.exit(1)
    
    # Sanitize sequences
    print("\n1. Sanitizing sequences...")
    convert_and_remove_unrelated_sequences(str(fasta_path), sequence_type)
    sanitized_path = f"{str(fasta_path)}.{sequence_type}.sanitized"
    print(f"   Sanitized file: {sanitized_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate matrix images for SSIM variants
    print("\n2. Generating self-comparison matrix images...")
    image_dir = output_dir / "matrices"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(sanitized_path, "r") as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
            print(f"   Generating images for {len(sequences)} sequences...")
            
            to_run = []
            for s in sequences:
                # Use description to match what the variant reads
                seq_name = s.description
                to_run.append((
                    seq_name, seq_name, s.seq, s.seq, 
                    255, str(image_dir), sequence_type
                ))
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                futures = [pool.submit(save_image_by_matrices, *data) for data in to_run]
                for i, f in enumerate(futures):
                    try:
                        f.result()
                        print(f"   Generated image for {sequences[i].description}")
                    except Exception as e:
                        print(f"   Warning: Failed to generate image for {sequences[i].description}: {e}")
            
            # Verify images were created
            # Images are saved in subdirectories (full, red, green, blue, etc.)
            # Use the 'full' subdirectory for SSIM analysis (contains complete RGB data)
            created_images = list((image_dir / "full").glob("*.png"))
            print(f"   ✓ Generated {len(created_images)} matrix images in {image_dir}/full")
            
            if len(created_images) != len(sequences):
                print(f"   Warning: Expected {len(sequences)} images but only created {len(created_images)}")
            
            # Update image_dir to point to the 'full' subdirectory
            image_dir = image_dir / "full"
                
    except Exception as e:
        import traceback
        print(f"   Warning: Could not generate images: {e}")
        traceback.print_exc()
        print(f"   Skipping SSIM variants...")
        image_dir = None
    
    # Run multiple variants
    print("\n3. Running analysis variants...")
    variants = []
    
    try:
        print("   - Needleman-Wunsch (Global Alignment)")
        nw = NeedlemanWunschVariant(sanitized_path, sequence_type)
        variants.append(nw)
    except Exception as e:
        import traceback
        print(f"   Warning: Could not create NW variant: {e}")
        traceback.print_exc()
    
    try:
        print("   - Smith-Waterman (Local Alignment)")
        sw = SmithWatermanVariant(sanitized_path, sequence_type)
        variants.append(sw)
    except Exception as e:
        import traceback
        print(f"   Warning: Could not create SW variant: {e}")
        traceback.print_exc()
    
    # Add SSIM variants if images were generated successfully
    if image_dir and image_dir.exists() and len(list(image_dir.glob("*.png"))) > 0:
        try:
            print("   - Resized SSIM Multi-Scale (Image-based)")
            rssim = ResizedSSIMMultiScaleVariant(sanitized_path, sequence_type, str(image_dir))
            variants.append(rssim)
        except Exception as e:
            import traceback
            print(f"   Warning: Could not create RSSIM variant: {e}")
            traceback.print_exc()
        
        try:
            print("   - Unrestricted SSIM (Image-based)")
            ussim = UnrestrictedSSIMVariant(sanitized_path, sequence_type, str(image_dir))
            variants.append(ussim)
        except Exception as e:
            import traceback
            print(f"   Warning: Could not create USSIM variant: {e}")
            traceback.print_exc()
        
        try:
            print("   - Greedy SSIM (Image-based)")
            gssim = GreedySSIMVariant(sanitized_path, sequence_type, str(image_dir))
            variants.append(gssim)
        except Exception as e:
            import traceback
            print(f"   Warning: Could not create GSSIM variant: {e}")
            traceback.print_exc()
        
        try:
            print("   - Windowed Multi-Scale SSIM (Image-based)")
            wmsssim = WindowedSSIMMultiScaleVariant(sanitized_path, sequence_type, str(image_dir))
            variants.append(wmsssim)
        except Exception as e:
            import traceback
            print(f"   Warning: Could not create WMSSSIM variant: {e}")
            traceback.print_exc()
        
        try:
            print("   - Deep Search (VGG16 + Annoy)")
            deep = DeepSearchVariant(sanitized_path, sequence_type, str(image_dir))
            variants.append(deep)
        except Exception as e:
            import traceback
            print(f"   Warning: Could not create Deep Search variant: {e}")
            traceback.print_exc()
    else:
        print("   Skipping SSIM variants (no images available)")
    
    if not variants:
        print("Error: No variants available for analysis")
        return
    
    # Run experiment
    print("\n4. Building distance matrices and phylogenetic trees...")
    experiment = Experiment(output_dir, *variants)
    experiment.run_and_save()
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    tree_paths = []
    for variant in variants:
        variant_name = variant.__class__.__name__.replace("Variant", "")
        print(f"  - {variant_name}.csv (distance matrix)")
        print(f"  - {variant_name}.nw (phylogenetic tree)")
        print(f"  - {variant_name}.png (tree visualization)")
    
    # Find all generated .nw files (excluding true_tree.nw)
    all_nw_files = sorted(output_dir.glob("*.nw"))
    tree_paths = [f for f in all_nw_files if f.name != "true_tree.nw"]
    
    return tree_paths


def main():
    parser = argparse.ArgumentParser(
        description="Evolve protein sequences and analyze with biomodelml"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('output_evolution_demo'),
        help='Output directory for results (default: output_evolution_demo)'
    )
    parser.add_argument(
        '--length', '-l',
        type=int,
        default=250,
        help='Length of ancestral sequence (default: 250)'
    )
    parser.add_argument(
        '--mutation-rate', '-m',
        type=float,
        default=0.002,
        help='Mutation rate per site per branch unit (default: 0.002)'
    )
    parser.add_argument(
        '--indel-rate', '-i',
        type=float,
        default=0.0005,
        help='Indel rate per site per branch unit (default: 0.0005)'
    )
    parser.add_argument(
        '--selection', '-s',
        type=float,
        default=0.3,
        help='Selection strength 0-1 (default: 0.3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num-taxa', '-n',
        type=int,
        default=7,
        help='Number of taxa (sequences) to generate (default: 7)'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip biomodelml analysis (just generate sequences)'
    )
    parser.add_argument(
        '--skip-comparison',
        action='store_true',
        help='Skip automatic tree comparison'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("EVOLUTIONARY SEQUENCE ANALYSIS DEMONSTRATION")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Number of taxa: {args.num_taxa}")
    print(f"  Ancestral sequence length: {args.length} amino acids")
    print(f"  Mutation rate: {args.mutation_rate} per site per branch unit")
    print(f"  Indel rate: {args.indel_rate} per site per branch unit")
    print(f"  Selection strength: {args.selection}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output}")
    
    # Step 1: Generate ancestral sequence
    print(f"\n{'='*60}")
    print("Step 1: Generating ancestral sequence")
    print(f"{'='*60}")
    ancestral_seq = generate_ancestral_sequence(args.length, args.seed)
    print(f"Generated ancestral sequence: {ancestral_seq[:50]}... ({len(ancestral_seq)} AA)")
    
    # Step 2: Build phylogenetic tree
    print(f"\n{'='*60}")
    print("Step 2: Building phylogenetic tree")
    print(f"{'='*60}")
    if args.num_taxa == 7:
        # Use predefined 7-taxa tree
        tree_root = build_example_tree()
    else:
        # Build custom tree with n taxa
        tree_root = build_tree_with_n_taxa(args.num_taxa, args.seed)
    tree_root.sequence = ancestral_seq
    
    leaves = tree_root.get_all_leaves()
    print(f"Tree structure: {len(leaves)} taxa")
    print(f"Tree topology (Newick): {tree_to_newick(tree_root, include_inner_labels=False)}")
    
    # Step 3: Evolve sequences
    print(f"\n{'='*60}")
    print("Step 3: Evolving sequences along tree")
    print(f"{'='*60}")
    evolver = SequenceEvolver(
        mutation_rate=args.mutation_rate,
        indel_rate=args.indel_rate,
        selection_strength=args.selection,
        seed=args.seed
    )
    evolve_along_tree(tree_root, evolver)
    
    # Print evolution summary
    for leaf in leaves:
        n_substitutions = sum(1 for m in leaf.mutations if m.event_type == 'substitution')
        n_insertions = sum(1 for m in leaf.mutations if m.event_type == 'insertion')
        n_deletions = sum(1 for m in leaf.mutations if m.event_type == 'deletion')
        print(f"  {leaf.name}: {len(leaf.sequence)} AA, "
              f"{n_substitutions} substitutions, "
              f"{n_insertions} insertions, "
              f"{n_deletions} deletions")
    
    # Step 4: Save sequences and metadata
    print(f"\n{'='*60}")
    print("Step 4: Saving sequences and metadata")
    print(f"{'='*60}")
    
    fasta_path = args.output / "evolved_sequences.fasta"
    save_sequences_fasta(leaves, fasta_path)
    print(f"  ✓ FASTA file: {fasta_path}")
    
    true_tree_path = args.output / "true_tree.nw"
    with open(true_tree_path, 'w') as f:
        f.write(tree_to_newick(tree_root, include_inner_labels=True))
    print(f"  ✓ True phylogeny: {true_tree_path}")
    
    mutation_history_path = args.output / "evolution_history.json"
    save_mutation_history(tree_root, mutation_history_path)
    print(f"  ✓ Mutation history: {mutation_history_path}")
    
    # Step 5: Run biomodelml analysis
    inferred_tree_paths = None
    if not args.skip_analysis:
        inferred_tree_paths = run_biomodelml_analysis(fasta_path, args.output, sequence_type="P")
    else:
        print("\n(Skipping biomodelml analysis as requested)")
    
    # Step 6: Compare trees and generate RF distance report
    if not args.skip_analysis and not args.skip_comparison and inferred_tree_paths:
        print(f"\n{'='*60}")
        print("Step 6: Comparing phylogenetic trees")
        print(f"{'='*60}")
        
        # Run tree comparison script
        comparison_script = Path(__file__).parent / "tree_comparison.py"
        if comparison_script.exists():
            try:
                # Build command with all inferred trees
                existing_trees = [str(p) for p in inferred_tree_paths if p.exists()]
                if not existing_trees:
                    print("   No inferred trees found to compare")
                    return
                
                cmd = [
                    sys.executable,
                    str(comparison_script),
                    str(true_tree_path),
                    *existing_trees,
                    "--output", str(args.output / "comparison_report.txt"),
                    "--visualize"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("\n✓ Tree comparison complete!")
                    
                    # Display Robinson-Foulds distance table
                    print("\n" + "="*60)
                    print("ROBINSON-FOULDS DISTANCE COMPARISON")
                    print("="*60)
                    print(f"{'Algorithm':<45} {'RF Distance':<12} {'Status'}")
                    print("-"*60)
                    
                    for tree_path in inferred_tree_paths:
                        if tree_path.exists():
                            algo_name = tree_path.stem
                            # Parse RF distance from comparison report if available
                            report_path = args.output / "comparison_report.txt"
                            if report_path.exists():
                                with open(report_path, 'r') as f:
                                    report_text = f.read()
                                    # Extract RF for this algorithm
                                    import re
                                    pattern = rf"{re.escape(algo_name)}.*?RF distance:\s*(\d+)"
                                    match = re.search(pattern, report_text, re.IGNORECASE)
                                    if match:
                                        rf_dist = match.group(1)
                                        status = "✓ Perfect" if rf_dist == "0" else ("✓ Good" if int(rf_dist) <= 2 else "⚠ High")
                                        print(f"{algo_name:<45} {rf_dist:<12} {status}")
                    
                    print("="*60)
                    print(f"\nDetailed report saved to: {args.output / 'comparison_report.txt'}")
                    print(f"Tree visualizations saved to: {args.output}/*_visualization.png")
                else:
                    print(f"\nWarning: Tree comparison failed: {result.stderr}")
                    
            except Exception as e:
                print(f"\nWarning: Could not run tree comparison: {e}")
        else:
            print(f"\nNote: Tree comparison script not found: {comparison_script}")
            print("Run manually: python tests/tree_comparison.py [trees...]")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {args.output}/")
    print("\nNext steps:")
    print(f"  1. View the true phylogeny: cat {true_tree_path}")
    print(f"  2. Compare with reconstructed trees: ls {args.output}/*.nw")
    print(f"  3. View distance matrices: ls {args.output}/*.csv")
    print(f"  4. Check mutation history: cat {mutation_history_path}")
    print("\nTo run tree comparison:")
    print(f"  python tests/tree_comparison.py {true_tree_path} {args.output}/*.nw")


if __name__ == '__main__':
    main()
