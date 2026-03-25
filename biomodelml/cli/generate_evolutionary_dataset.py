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

import sys
import os

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomodelml.matrices import build_matrix


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


def evolve_tree_pyvolve(root: TreeNode, seq_len: int, alphabet: str = 'ACGT') -> None:
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

    # Set up evolution model (HKY85 for DNA)
    model = pyvolve.Model("nucleotide", {"kappa": 4.0})
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

    def assign_sequences(node):
        if node.name in evolved_seqs:
            node.sequence = evolved_seqs[node.name]
        for child in node.children:
            assign_sequences(child)

    assign_sequences(root)


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
            evolve_tree_pyvolve(root, args.seq_len, alphabet='ACGT')

            # Get all leaf nodes (taxa)
            leaves = root.get_all_leaves()

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

    # Output parameters
    parser.add_argument('--output-path', type=str, default='data/evolutionary_10k.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate
    if args.n_taxa < 2:
        print("Error: n_taxa must be at least 2")
        sys.exit(1)

    generate_dataset(args)
