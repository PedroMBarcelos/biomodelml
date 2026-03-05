# Evolutionary Sequence Analysis Demo

This demonstration shows how to simulate protein sequence evolution along a known phylogenetic tree, then use biomodelml to reconstruct the phylogeny from the sequences.

## Overview

The demo consists of three main components:

1. **`demo_evolution_analysis.py`** - Main script that:
   - Generates an ancestral protein sequence
   - Builds a phylogenetic tree with 8 taxa
   - Simulates evolution along branches with mutations, indels, and selection
   - Saves evolved sequences to FASTA format
   - Runs biomodelml analysis with multiple algorithms
   - Saves all results and metadata

2. **`tree_comparison.py`** - Utility to compare phylogenies:
   - Calculates Robinson-Foulds distance (topological similarity)
   - Computes branch length correlation
   - Identifies which algorithms best recovered the true tree
   - Generates comparison reports

3. **Validation** - Assess biomodelml's phylogenetic reconstruction accuracy

## Quick Start

### Basic Usage

```bash
# Run the complete demo (evolution + analysis)
python tests/demo_evolution_analysis.py

# Results will be in output_evolution_demo/
```

### Custom Parameters

```bash
# Custom output directory and parameters
python tests/demo_evolution_analysis.py \
    --output my_evolution_test \
    --length 300 \
    --mutation-rate 0.003 \
    --indel-rate 0.001 \
    --selection 0.5 \
    --seed 12345
```

### Compare Results

```bash
# Compare reconstructed trees with the true tree
python tests/tree_comparison.py \
    output_evolution_demo/true_tree.nw \
    output_evolution_demo/*.nw
```

## Output Files

After running the demo, you'll find these files in the output directory:

### Generated Sequences
- **`evolved_sequences.fasta`** - The evolved protein sequences in FASTA format
- **`evolved_sequences.fasta.P.sanitized`** - Sanitized version used for analysis
- **`true_tree.nw`** - The true phylogenetic tree in Newick format
- **`evolution_history.json`** - Complete mutation history for all sequences

### Analysis Results (per algorithm)
Each algorithm generates:
- **`AlgorithmName.csv`** - Distance matrix (sequence-by-sequence)
- **`AlgorithmName.nw`** - Reconstructed phylogenetic tree (Newick format)
- **`AlgorithmName.png`** - Tree visualization

Algorithms included:
- `NeedlemanWunsch` - Global sequence alignment (traditional)
- `ResizedSSIMMultiScale` - Multi-scale SSIM with image resizing (novel)
- `UnrestrictedSSIM` - SSIM with dynamic programming (novel)

### Comparison Results
- **`comparison_report.txt`** - Detailed comparison metrics (if generated)

## Parameters Explained

### Sequence Parameters

- **`--length`** (default: 250)
  - Length of the ancestral protein sequence in amino acids
  - Longer sequences provide more data but slower evolution
  - Recommended: 200-500 for meaningful phylogenetic signal

### Evolution Parameters

- **`--mutation-rate`** (default: 0.002)
  - Probability of point mutation per site per branch length unit
  - Example: 0.002 means ~0.2% of sites mutate per unit branch length
  - Higher values = faster evolution, more divergence
  - Recommended range: 0.001-0.005

- **`--indel-rate`** (default: 0.0005)
  - Probability of insertion/deletion per site per branch length unit
  - Indels length: 1-5 amino acids
  - Lower than mutation rate (indels are rarer in reality)
  - Recommended range: 0.0001-0.001

- **`--selection`** (default: 0.3)
  - Strength of purifying selection (0.0 = neutral, 1.0 = strong)
  - Penalizes drastic amino acid changes (e.g., hydrophobic ↔ charged)
  - Mimics functional constraints on protein evolution
  - Recommended range: 0.2-0.5 for realistic evolution

### Other Parameters

- **`--seed`** (default: 42)
  - Random seed for reproducibility
  - Use the same seed to get identical results
  
- **`--skip-analysis`**
  - Generate sequences only, don't run biomodelml
  - Useful for testing sequence generation separately

- **`--output`** (default: output_evolution_demo)
  - Directory where all results are saved

## Understanding the Tree Structure

The demo generates an 8-taxa tree with this structure:

```
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
```

**Sister taxa** (expected to be closely related):
- Seq2 & Seq3 (share Node_C)
- Seq7 & Seq8 (share Node_E)
- Seq6 & (Seq7+Seq8) (share Node_D)

**Major clades**:
- Left clade: Seq1, Seq2, Seq3
- Right clade: Seq5, Seq6, Seq7, Seq8

### Branch Lengths

Branch lengths represent evolutionary time/distance:
- Short branches (0.5-0.8) → less divergence → more similar sequences
- Long branches (1.0-1.5) → more divergence → less similar sequences

## Interpreting Results

### Evolution Summary

The script prints evolution statistics for each sequence:
```
Seq1: 245 AA, 12 substitutions, 1 insertions, 2 deletions
```
- **AA count**: Final sequence length (may differ from ancestral due to indels)
- **Substitutions**: Point mutations (amino acid changes)
- **Insertions**: Added amino acids
- **Deletions**: Removed amino acids

### Tree Comparison Metrics

When running `tree_comparison.py`, you'll see:

#### Robinson-Foulds Distance (RF)
- Measures topological difference between trees
- **RF = 0**: Trees have identical topology (perfect!)
- **RF = 2-4**: Minor differences (good)
- **RF > 6**: Major topological differences

#### Normalized RF Distance
- RF distance scaled to [0, 1]
- **0.0**: Perfect topology match
- **< 0.3**: Good reconstruction
- **> 0.5**: Poor reconstruction

#### Branch Length Correlation
- Pearson correlation of branch lengths for common branches
- **> 0.8**: Excellent correlation
- **0.5-0.8**: Good correlation
- **< 0.5**: Poor correlation

#### Topology Match
- ✓ if RF distance = 0 (exact match)
- ✗ if any topological differences

### Success Criteria

A successful phylogenetic reconstruction should have:
1. **RF distance < 4** (at least one algorithm)
2. **Monophyly preserved** for sister taxa
3. **Branch length correlation > 0.6** (at least one algorithm)

## Examples

### Fast Evolution (High Divergence)

```bash
python tests/demo_evolution_analysis.py \
    --mutation-rate 0.005 \
    --indel-rate 0.001 \
    --selection 0.2
```
- Sequences will be more divergent
- Harder to reconstruct accurate phylogeny
- Tests algorithm robustness

### Slow Evolution (Low Divergence)

```bash
python tests/demo_evolution_analysis.py \
    --mutation-rate 0.001 \
    --indel-rate 0.0002 \
    --selection 0.5
```
- Sequences will be more similar
- Easier to reconstruct phylogeny
- Less phylogenetic signal

### Large Protein

```bash
python tests/demo_evolution_analysis.py \
    --length 500 \
    --mutation-rate 0.002
```
- More data = better phylogenetic signal
- Longer computation time
- More realistic for actual proteins

### Neutral Evolution (No Selection)

```bash
python tests/demo_evolution_analysis.py \
    --selection 0.0
```
- All mutations accepted regardless of amino acid properties
- Faster sequence divergence
- Less biologically realistic

## Advanced Usage

### Using Evolution History

The `evolution_history.json` file contains complete mutation records:

```python
import json

with open('output_evolution_demo/evolution_history.json') as f:
    history = json.load(f)

# Access mutation events
for mutation in history['children'][0]['mutations']:
    print(f"{mutation['type']} at position {mutation['position']}")
```

### Analyzing Distance Matrices

```python
import pandas as pd

# Load distance matrix
df = pd.read_csv('output_evolution_demo/NeedlemanWunsch.csv', index_col=0)

# Most similar sequences
min_dist = df.where(df > 0).min().min()
print(f"Most similar pair: {min_dist}")

# Most divergent sequences
max_dist = df.max().max()
print(f"Most divergent pair: {max_dist}")
```

### Parsing Newick Trees

```python
from ete3 import Tree

# Load reconstructed tree
tree = Tree('output_evolution_demo/ResizedSSIMMultiScale.nw', format=1)

# Print tree structure
print(tree)

# Get all leaf names
leaves = tree.get_leaf_names()
print(f"Sequences: {leaves}")

# Visualize (requires X server or virtual display)
tree.render('my_tree.png')
```

## Troubleshooting

### Import Errors

If you see `ImportError: No module named 'biomodelml'`:
```bash
# Install biomodelml in development mode
cd /path/to/biomodelml
pip install -e .
```

### Missing ete3

If tree comparison fails:
```bash
pip install ete3
```

### Very Short Sequences

If sequences become very short after evolution (excessive deletions):
- Reduce `--indel-rate`
- Increase `--selection` (selection against indels)
- Increase `--length` for larger starting sequence

### No Phylogenetic Signal

If all distance matrices show similar values:
- Increase `--mutation-rate` for more divergence
- Check that branch lengths are reasonable (not all tiny)
- Try longer sequences (`--length`)

## Validation Strategy

To validate biomodelml's performance:

1. **Run multiple replicates** with different seeds
2. **Vary evolutionary parameters** to test different scenarios
3. **Compare algorithms** to see which works best for your data
4. **Check monophyly** of known sister groups
5. **Examine distance correlations** with true evolutionary distances

### Batch Testing

```bash
#!/bin/bash
# Run 10 replicates
for seed in {1..10}; do
    python tests/demo_evolution_analysis.py \
        --output output_replicate_${seed} \
        --seed ${seed}
    
    python tests/tree_comparison.py \
        output_replicate_${seed}/true_tree.nw \
        output_replicate_${seed}/*.nw \
        --output output_replicate_${seed}/comparison.txt
done

# Aggregate results
grep "Normalized RF" output_replicate_*/comparison.txt
```

## Citation

If you use this demo in your research or teaching, please cite biomodelml:

```
[Add appropriate citation when available]
```

## Questions?

For issues, questions, or suggestions:
- Open an issue on GitHub: https://github.com/BioBD/biomodelml/issues
- Check the main README: ../README.md
- Review the biomodelml documentation

## License

This demo is part of biomodelml and is distributed under the MIT License.
See LICENSE file for details.
