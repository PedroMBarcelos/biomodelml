"""
Generate Evolutionary Dataset from Phylogenetic Trees

This script generates training data for the 4-Channel Siamese regressor using
realistic phylogenetic trees. Unlike on-the-fly generation with random parent-child
pairs, this approach:

1. Builds phylogenetic trees with N taxa
2. Evolves sequences down the tree using Pyvolve
3. Extracts all (N choose 2) pairwise distances from tree topology
4. Generates 4-channel matrices (RGB + Mask) for all sequence pairs
5. Saves to HDF5 format for fast training

This produces more realistic distance distributions since sequences are related
through a proper phylogenetic tree rather than random divergence times.
"""

import argparse
import h5py
import numpy as np
import random
import pyvolve
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import sys
import os

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomodelml.matrices import build_matrix


DEFAULT_GTR_RATES = (1.0, 2.0, 1.0, 1.0, 2.0, 1.0)
DEFAULT_BASE_FREQS = (0.25, 0.25, 0.25, 0.25)


class TreeNode:
    """
    Simple tree node for phylogenetic trees.

    Attributes:
        name (str): Node identifier
        sequence (str): DNA/protein sequence at this node
        branch_length (float): Branch length from parent to this node
        parent (TreeNode): Parent node (None for root)
        children (List[TreeNode]): List of child nodes
    """
    def __init__(self, name: str):
        self.name = name
        self.sequence = None
        self.branch_length = 0.0
        self.parent = None
        self.children = []

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (taxon)"""
        return len(self.children) == 0

    def get_all_leaves(self) -> List['TreeNode']:
        """Get all leaf nodes (taxa) under this node"""
        if self.is_leaf():
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves


def build_random_tree(num_taxa: int, min_branch: float = 0.01,
                     max_branch: float = 0.5, seed: Optional[int] = None) -> TreeNode:
    """
    Build a random phylogenetic tree with N taxa.

    Creates a balanced binary tree structure with random branch lengths.

    Args:
        num_taxa: Number of leaf nodes (taxa)
        min_branch: Minimum branch length
        max_branch: Maximum branch length
        seed: Random seed for reproducibility

    Returns:
        Root node of the phylogenetic tree
    """
    if seed is not None:
        random.seed(seed)

    if num_taxa < 2:
        raise ValueError("Need at least 2 taxa")

    # Create leaf nodes
    leaves = [TreeNode(f"Taxon_{i+1}") for i in range(num_taxa)]
    nodes = leaves[:]

    # Build tree bottom-up by pairing nodes
    node_counter = 0
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                # Create parent for pair
                parent = TreeNode(f"inner_{node_counter}")
                node_counter += 1
                child1, child2 = nodes[i], nodes[i+1]

                # Assign random branch lengths
                child1.branch_length = random.uniform(min_branch, max_branch)
                child2.branch_length = random.uniform(min_branch, max_branch)
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


def tree_to_pyvolve_string(node: TreeNode) -> str:
    """
    Convert tree to Pyvolve-compatible Newick string.

    Pyvolve requires ALL nodes (including root) to have branch lengths.
    Internal node labels are omitted as Pyvolve may not handle them well.

    Args:
        node: Root node of the tree

    Returns:
        Newick format string (e.g., "((A:0.1,B:0.2):0.15,(C:0.3,D:0.4):0.25):0;")
    """
    if node.is_leaf():
        return f"{node.name}:{node.branch_length:.6f}"

    children_newick = ','.join(tree_to_pyvolve_string(child) for child in node.children)

    if node.parent is None:  # Root
        # Root needs a branch length (set to 0.0) but no label
        return f"({children_newick}):0.0;"
    else:
        # Internal nodes: no label, just branch length
        return f"({children_newick}):{node.branch_length:.6f}"


def get_path_to_root(node: TreeNode) -> List[TreeNode]:
    """Get the path from a node to the root"""
    path = []
    current = node
    while current is not None:
        path.append(current)
        current = current.parent
    return path


def calculate_tree_distance(node1: TreeNode, node2: TreeNode) -> float:
    """
    Calculate evolutionary distance between two nodes.

    Returns the sum of branch lengths along the path from node1 to node2
    through their most recent common ancestor (MRCA).

    Args:
        node1: First node
        node2: Second node

    Returns:
        Evolutionary distance (sum of branch lengths)
    """
    if node1 == node2:
        return 0.0

    # Get paths to root
    path1 = get_path_to_root(node1)
    path2 = get_path_to_root(node2)

    # Find MRCA
    path1_ids = {id(node): node for node in path1}
    mrca = None
    for node in path2:
        if id(node) in path1_ids:
            mrca = node
            break

    if mrca is None:
        raise ValueError("Nodes are not in the same tree")

    # Calculate distance: sum of branch lengths from node1 to MRCA and node2 to MRCA
    distance = 0.0

    # Distance from node1 to MRCA
    current = node1
    while current != mrca:
        distance += current.branch_length
        current = current.parent

    # Distance from node2 to MRCA
    current = node2
    while current != mrca:
        distance += current.branch_length
        current = current.parent

    return distance


def evolve_tree_pyvolve(
    root: TreeNode,
    seq_len: int,
    alphabet: str = 'ACGT',
    model_name: str = 'gtr',
    kappa: float = 4.0,
    gtr_rates: Tuple[float, float, float, float, float, float] = DEFAULT_GTR_RATES,
    base_freqs: Tuple[float, float, float, float] = DEFAULT_BASE_FREQS,
    gamma_alpha: float = 0.5,
    gamma_categories: int = 4,
    p_invariant: float = 0.1,
    indel_rate: float = 0.0005,
    indel_size: int = 1,
    max_len: Optional[int] = None
) -> None:
    """
    Evolve sequences down the tree using Pyvolve.

    This modifies the tree in-place, setting the sequence attribute for all nodes.

    Args:
        root: Root node (will be assigned random ancestral sequence)
        seq_len: Length of sequences to evolve
        alphabet: Sequence alphabet (default: 'ACGT' for DNA)
    """
    # Generate random ancestral sequence
    root.sequence = "".join(random.choice(alphabet) for _ in range(seq_len))

    # Build Newick string
    newick_str = tree_to_pyvolve_string(root)

    # Parse tree with Pyvolve
    pyv_tree = pyvolve.read_tree(tree=newick_str)

    # Set up evolution model (GTR+G+I default, HKY85 optional)
    model_name = model_name.lower()
    if model_name == 'hky85':
        model_params = {
            "kappa": float(kappa),
            "state_freqs": list(base_freqs),
            "alpha": float(gamma_alpha),
            "num_categories": int(gamma_categories),
            "pinv": float(p_invariant),
        }
    elif model_name == 'gtr':
        model_params = {
            "mu": {
                "AC": float(gtr_rates[0]),
                "AG": float(gtr_rates[1]),
                "AT": float(gtr_rates[2]),
                "CG": float(gtr_rates[3]),
                "CT": float(gtr_rates[4]),
                "GT": float(gtr_rates[5]),
            },
            "state_freqs": list(base_freqs),
            "alpha": float(gamma_alpha),
            "num_categories": int(gamma_categories),
            "pinv": float(p_invariant),
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Use 'gtr' or 'hky85'.")

    model = pyvolve.Model("nucleotide", model_params)
    indel_supported = False
    if indel_rate > 0:
        try:
            partition = pyvolve.Partition(
                models=model,
                size=seq_len,
                indel_rate=float(indel_rate),
                indel_size=max(1, int(indel_size)),
            )
            indel_supported = True
        except TypeError:
            print("Warning: This Pyvolve version does not expose indel Partition options; running substitution-only evolution.")
            partition = pyvolve.Partition(models=model, size=seq_len)
    else:
        partition = pyvolve.Partition(models=model, size=seq_len)

    # Evolve
    evolver = pyvolve.Evolver(
        tree=pyv_tree,
        partitions=partition,
        custom_seqs={root.name: root.sequence}
    )
    evolver(seqfile=None,ratefile=None, infofile=None)

    # Extract sequences and assign to nodes
    evolved_seqs = evolver.get_sequences(anc=True)  # Get all sequences including ancestors

    def apply_fallback_indels(seq: str) -> str:
        if indel_rate <= 0:
            return seq

        seq_list = list(seq)
        expected_indels = int(max(0, len(seq_list) * indel_rate))
        n_events = max(0, int(random.gauss(expected_indels, (expected_indels + 1) ** 0.5)))

        for _ in range(n_events):
            if not seq_list:
                break
            is_deletion = random.random() < 0.5 and len(seq_list) > 10
            if is_deletion:
                del_size = min(max(1, int(indel_size)), len(seq_list) - 1)
                if del_size <= 0:
                    continue
                pos = random.randint(0, len(seq_list) - del_size)
                del seq_list[pos:pos + del_size]
            else:
                ins_size = max(1, int(indel_size))
                pos = random.randint(0, len(seq_list))
                ins = [random.choice(alphabet) for _ in range(ins_size)]
                seq_list[pos:pos] = ins

        return ''.join(seq_list)

    def assign_sequences(node):
        if node.name in evolved_seqs:
            seq = evolved_seqs[node.name]
            if indel_rate > 0 and not indel_supported:
                seq = apply_fallback_indels(seq)
            if max_len is not None:
                seq = seq[:max_len]
            node.sequence = seq
        for child in node.children:
            assign_sequences(child)

    assign_sequences(root)


def save_tree_sequences_fasta(leaves: List[TreeNode], tree_idx: int,
                               output_path: Path) -> None:
    """
    Save leaf sequences from a tree to FASTA file using BioPython.

    Creates a FASTA file with one sequence per leaf node, including metadata
    in headers for tree index, branch length, and leaf position.

    Args:
        leaves: List of leaf nodes with sequences
        tree_idx: Index of the tree in the dataset
        output_path: Base output path (will create _tree_XXXX.fasta files)
    """
    # Construct FASTA filename
    base_name = output_path.stem  # e.g., "evolutionary_10k"
    fasta_path = output_path.parent / f"{base_name}_tree_{tree_idx:04d}.fasta"

    # Create SeqRecord objects with metadata
    records = []
    for leaf_idx, leaf in enumerate(leaves):
        # Build metadata-rich header
        record_id = leaf.name
        description = f"tree={tree_idx} branch_len={leaf.branch_length:.6f} leaf_idx={leaf_idx}"

        record = SeqRecord(
            Seq(leaf.sequence),
            id=record_id,
            description=description
        )
        records.append(record)

    # Write to FASTA file
    SeqIO.write(records, fasta_path, "fasta")


def process_sequence_4ch(seq: str, max_len: int, seq_type: str = 'N') -> np.ndarray:
    """
    Process a sequence into a 4-channel tensor (RGB + Mask).

    Args:
        seq: DNA or protein sequence string
        max_len: Maximum length for padding
        seq_type: 'N' for nucleotide, 'P' for protein

    Returns:
        4-channel tensor of shape (4, max_len, max_len) as float32
    """
    # Keep fixed-size tensors valid even if indels increase sequence length.
    if len(seq) > max_len:
        seq = seq[:max_len]

    # Build self-comparison matrix
    seq_obj = Seq(seq)
    matrix = build_matrix(seq_obj, seq_obj, max_len, seq_type=seq_type)
    h, w, c = matrix.shape

    # Convert to tensor
    rgb_tensor = torch.from_numpy(matrix).permute(2, 0, 1).float() / 255.0  # (3, H, W) [0, 1]

    # Create mask
    mask = np.ones((h, w), dtype=np.float32)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W) [0, 1]

    # Pad
    pad_h = max_len - h
    pad_w = max_len - w
    rgb_padded = F.pad(rgb_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    mask_padded = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Normalize RGB to [-1, 1], keep mask at [0, 1]
    mean_rgb = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std_rgb = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    rgb_normalized = (rgb_padded - mean_rgb) / std_rgb

    # Concatenate to 4 channels
    tensor_4ch = torch.cat([rgb_normalized, mask_padded], dim=0)  # (4, max_len, max_len)

    return tensor_4ch.numpy().astype(np.float32)


def generate_dataset(args):
    """
    Generate the complete phylogenetic dataset.

    For each tree:
        1. Build random phylogenetic tree with N taxa
        2. Evolve sequences down the tree
        3. Extract all (N choose 2) pairwise combinations
        4. Calculate tree-based distance for each pair
        5. Generate 4-channel matrices
        6. Append to HDF5 file

    Args:
        args: Command-line arguments
    """
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate total number of pairs
    pairs_per_tree = args.n_taxa * (args.n_taxa - 1) // 2
    total_pairs = pairs_per_tree * args.num_trees

    print("=" * 70)
    print("Phylogenetic Dataset Generation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Number of taxa:        {args.n_taxa}")
    print(f"  Number of trees:       {args.num_trees}")
    print(f"  Sequence length:       {args.seq_len}")
    print(f"  Evolution model:       {args.model_name.upper()}+G+I")
    print(f"  Indel rate:            {args.indel_rate}")
    print(f"  Matrix size:           {args.max_len}×{args.max_len}")
    print(f"  Pairs per tree:        {pairs_per_tree}")
    print(f"  Total pairs:           {total_pairs}")
    print(f"  Expected file size:    ~{total_pairs * args.max_len * args.max_len * 4 * 4 * 2 / 1e9:.2f} GB")
    print("=" * 70)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        # Pre-allocate datasets (will grow dynamically)
        img1_dset = h5f.create_dataset(
            'img1',
            shape=(0, 4, args.max_len, args.max_len),
            maxshape=(None, 4, args.max_len, args.max_len),
            dtype='float32',
            compression='gzip',
            compression_opts=1  # Fast compression
        )

        img2_dset = h5f.create_dataset(
            'img2',
            shape=(0, 4, args.max_len, args.max_len),
            maxshape=(None, 4, args.max_len, args.max_len),
            dtype='float32',
            compression='gzip',
            compression_opts=1
        )

        dist_dset = h5f.create_dataset(
            'dist',
            shape=(0, 1),
            maxshape=(None, 1),
            dtype='float32'
        )

        # Store metadata
        h5f.attrs['n_taxa'] = args.n_taxa
        h5f.attrs['num_trees'] = args.num_trees
        h5f.attrs['seq_len'] = args.seq_len
        h5f.attrs['max_len'] = args.max_len
        h5f.attrs['seq_type'] = args.seq_type
        h5f.attrs['model_name'] = args.model_name
        h5f.attrs['kappa'] = args.kappa
        h5f.attrs['gtr_rates'] = np.asarray(args.gtr_rates, dtype=np.float32)
        h5f.attrs['base_freqs'] = np.asarray(args.base_freqs, dtype=np.float32)
        h5f.attrs['gamma_alpha'] = args.gamma_alpha
        h5f.attrs['gamma_categories'] = args.gamma_categories
        h5f.attrs['p_invariant'] = args.p_invariant
        h5f.attrs['indel_rate'] = args.indel_rate
        h5f.attrs['indel_size'] = args.indel_size

        current_idx = 0

        # Progress bar
        pbar = tqdm(total=total_pairs, desc="Generating pairs")

        for tree_idx in range(args.num_trees):
            # Build tree
            seed = args.seed + tree_idx if args.seed is not None else None
            root = build_random_tree(
                args.n_taxa,
                min_branch=args.min_branch,
                max_branch=args.max_branch,
                seed=seed
            )

            # Evolve sequences
            evolve_tree_pyvolve(
                root,
                args.seq_len,
                alphabet='ACGT',
                model_name=args.model_name,
                kappa=args.kappa,
                gtr_rates=tuple(args.gtr_rates),
                base_freqs=tuple(args.base_freqs),
                gamma_alpha=args.gamma_alpha,
                gamma_categories=args.gamma_categories,
                p_invariant=args.p_invariant,
                indel_rate=args.indel_rate,
                indel_size=args.indel_size,
                max_len=args.max_len,
            )

            # Get all leaf nodes (taxa)
            leaves = root.get_all_leaves()

            # Save sequences to FASTA if requested
            if args.save_fasta:
                save_tree_sequences_fasta(leaves, tree_idx, output_path)

            # Generate all pairwise combinations
            pairs_to_add = []

            for leaf1, leaf2 in combinations(leaves, 2):
                # Calculate tree distance
                distance = calculate_tree_distance(leaf1, leaf2)

                # Process sequences to 4-channel tensors
                matrix1 = process_sequence_4ch(leaf1.sequence, args.max_len, args.seq_type)
                matrix2 = process_sequence_4ch(leaf2.sequence, args.max_len, args.seq_type)

                pairs_to_add.append((matrix1, matrix2, distance))

            # Batch write to HDF5 (more efficient)
            num_new_pairs = len(pairs_to_add)
            new_size = current_idx + num_new_pairs

            # Resize datasets
            img1_dset.resize((new_size, 4, args.max_len, args.max_len))
            img2_dset.resize((new_size, 4, args.max_len, args.max_len))
            dist_dset.resize((new_size, 1))

            # Write batch
            for i, (mat1, mat2, dist) in enumerate(pairs_to_add):
                img1_dset[current_idx + i] = mat1
                img2_dset[current_idx + i] = mat2
                dist_dset[current_idx + i] = [dist]

            current_idx = new_size
            pbar.update(num_new_pairs)

        pbar.close()

    print("\n" + "=" * 70)
    print(f"Dataset generation complete!")
    print(f"Saved to: {output_path}")
    print(f"Total pairs: {current_idx}")
    if args.save_fasta:
        print(f"FASTA files: {args.num_trees} files in {output_path.parent}/")
    print("=" * 70)

    # Print distance statistics
    with h5py.File(output_path, 'r') as h5f:
        distances = h5f['dist'][:]
        print(f"\nDistance statistics:")
        print(f"  Min:    {distances.min():.6f}")
        print(f"  Max:    {distances.max():.6f}")
        print(f"  Mean:   {distances.mean():.6f}")
        print(f"  Median: {np.median(distances):.6f}")
        print(f"  Std:    {distances.std():.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate evolutionary dataset from phylogenetic trees"
    )

    # Tree parameters
    parser.add_argument('--n-taxa', type=int, default=10,
                        help='Number of taxa (leaf nodes) per tree')
    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees to generate')
    parser.add_argument('--min-branch', type=float, default=0.01,
                        help='Minimum branch length')
    parser.add_argument('--max-branch', type=float, default=0.5,
                        help='Maximum branch length')

    # Sequence parameters
    parser.add_argument('--seq-len', type=int, default=500,
                        help='Length of sequences')
    parser.add_argument('--max-len', type=int, default=550,
                        help='Matrix padding size')
    parser.add_argument('--seq-type', type=str, default='N', choices=['N', 'P'],
                        help='Sequence type: N=nucleotide, P=protein')

    # Evolution model parameters
    parser.add_argument('--model-name', type=str, default='gtr', choices=['gtr', 'hky85'],
                        help='Nucleotide substitution model')
    parser.add_argument('--kappa', type=float, default=4.0,
                        help='Transition/transversion ratio for HKY85 mode')
    parser.add_argument('--gtr-rates', type=float, nargs=6,
                        default=list(DEFAULT_GTR_RATES),
                        metavar=('AC', 'AG', 'AT', 'CG', 'CT', 'GT'),
                        help='Six exchangeability rates for GTR model')
    parser.add_argument('--base-freqs', type=float, nargs=4,
                        default=list(DEFAULT_BASE_FREQS),
                        metavar=('A', 'C', 'G', 'T'),
                        help='Base frequencies used by HKY85/GTR models')
    parser.add_argument('--gamma-alpha', type=float, default=0.5,
                        help='Alpha parameter for discrete gamma rate heterogeneity')
    parser.add_argument('--gamma-categories', type=int, default=4,
                        help='Number of discrete gamma rate categories')
    parser.add_argument('--p-invariant', type=float, default=0.1,
                        help='Proportion of invariant sites (I component)')
    parser.add_argument('--indel-rate', type=float, default=0.0005,
                        help='Indel rate for nucleotide evolution')
    parser.add_argument('--indel-size', type=int, default=1,
                        help='Default indel event size')

    # Output parameters
    parser.add_argument('--output-path', type=str, default='data/evolutionary_10k.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--save-fasta', action='store_true',
                        help='Save sequences to FASTA files (one per tree)')

    args = parser.parse_args()

    # Validate
    if args.n_taxa < 2:
        print("Error: n_taxa must be at least 2")
        sys.exit(1)

    if abs(sum(args.base_freqs) - 1.0) > 1e-6:
        print("Error: --base-freqs must sum to 1.0")
        sys.exit(1)

    if not (0.0 <= args.p_invariant < 1.0):
        print("Error: --p-invariant must be in [0.0, 1.0)")
        sys.exit(1)

    if args.gamma_categories < 1:
        print("Error: --gamma-categories must be >= 1")
        sys.exit(1)

    if args.indel_size < 1:
        print("Error: --indel-size must be >= 1")
        sys.exit(1)

    generate_dataset(args)
