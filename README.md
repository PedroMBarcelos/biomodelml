# BioModelML

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**BioModelML** is a foundational Python framework that bridges bioinformatics and computer vision. It transforms biological sequences (DNA, RNA, and proteins) into rich, spatially-encoded RGB matrices. 

By representing sequences as image tensors, BioModelML allows researchers to bypass traditional string-matching algorithms and directly apply state-of-the-art deep learning architectures to complex biological problems.

##  The Paradigm: Generate Once, Predict Anything

The core philosophy of BioModelML is **task-agnostic representation**. 

Instead of building separate string-processing pipelines for different biological questions, BioModelML standardizes the input. You convert your sequences into our specialized RGB matrices once, and then feed those standardized tensors into various downstream deep learning models depending on your objective:

* **Currently Implemented:** Phylogenetic tree reconstruction and evolutionary distance mapping.
* **Future Expansions:** 3D structure elucidation, functional prediction, and motif discovery. 

##  How It Works: The Encoding

BioModelML converts sequences into RGB image matrices where each pixel `(i, j)` encodes the specific biochemical and structural relationship between position `i` and `j`. 

**Nucleotide Sequences (DNA/RNA):**
- **Red Channel:** Self-comparison (matching positions)
- **Green Channel:** Complementary pairing (e.g., A↔T, C↔G)
- **Blue Channel:** Non-matching indicator

**Protein Sequences:**
- **Red Channel:** Substitution matrix scores (PROTSUB)
- **Green Channel:** Self-comparison (identity)
- **Blue Channel:** Sneath index scores (biochemical/hydrophobic similarity)

##  Installation

```bash
# Basic installation (Core encoding generation)
pip install biomodelml

# With all features (Deep learning, SSIM variants, optimized search)
pip install biomodelml[full]

# For GPU acceleration (Highly recommended for DL tasks)
pip install biomodelml[gpu]
```

### Optional: Data Generation Dependencies

To use the supervised training data generation pipeline (generating synthetic sequences with AliSim):

```bash
# On Ubuntu/Debian
sudo apt-get install iqtree

# On macOS (via Homebrew)
brew install iqtree

# Or download from https://www.iqtree.org/
```

##  Usage

### 1. The Core Pipeline: Sequence to Matrix

Generate image matrices programmatically to feed directly into your custom PyTorch DataLoader.

```bash
# Sanitize sequences
biomodelml-sanitize mysequences.fasta N

# Generate minimal, full-RGB matrices for an entire dataset
biomodelml-matchmatrix mysequences.fasta.N.sanitized output_tensors/ N
```

### 2. Supervised Training Data Generation

Generate synthetic evolutionary datasets for training deep learning models with known ground-truth phylogenetic distances using **AliSim** (integrated into IQ-TREE).

#### Prerequisites
First, install IQ-TREE with AliSim support:

```bash
# On Ubuntu/Debian
sudo apt-get install iqtree

# On macOS (via Homebrew)
brew install iqtree

# Or download from https://www.iqtree.org/
```

#### Step 1: Generate Synthetic Sequences

Use the `biomodelml-generate-sequences` CLI to create FASTA files with known evolutionary distances:

```bash
# Generate sequences using the "training" preset
biomodelml-generate-sequences output_dir/ \
  --preset training \
  --num-replicates 10 \
  --num-sequences 20 \
  --sequence-type N

# Or customize with overrides
biomodelml-generate-sequences output_dir/ \
  --preset benchmark \
  --alignment-length 2000 \
  --evolution-rate mixed \
  --tree-style random \
  --seed 42
```

**Available Presets:**
- **training**: 500bp, moderate evolution, GTR model, balanced tree (recommended for most use cases)
- **benchmark**: 1000bp, mixed evolution, HKY model, random tree (for diverse training data)
- **noise**: 300bp, high evolution, GTR+Gamma model, power-law tree (robust model training)

**Output Structure:**
```
output_dir/
├── sequences/
│   ├── replicate_001/
│   │   └── alignment.fasta
│   ├── replicate_002/
│   │   └── alignment.fasta
│   └── ...
├── trees/
│   ├── replicate_001/
│   │   └── tree.nwk
│   └── ...
└── metadata/
    ├── replicate_001/
    │   └── distances.csv
    └── ...
```

#### Step 2: Generate Image Matrices

Convert FASTA sequences to HDF5-backed RGB matrix shards:

```bash
biomodelml-generate-images output_dir/sequences/ images_output/ N \
  --max-window 255 \
  --num-workers 4 \
  --include-metadata

# For protein sequences
biomodelml-generate-images output_dir/sequences/ images_output/ P
```

**Output Structure:**
```
images_output/
├── images/
│   ├── replicate_001.h5
│   ├── replicate_002.h5
│   └── ...
└── metadata/
    └── image_manifest.json
```

Each HDF5 shard stores one dataset per generated matrix, plus checksums and sample metadata for validation.

#### Step 3: Create PyTorch Datasets

Use the Python API to load images with ground-truth distances for training:

```python
from biomodelml.data_generation import TrainingDataset

# Load the dataset
dataset = TrainingDataset("images_output/", lazy_load=True)

# Split into train/val/test with stratification by replicate
train_ds, val_ds, test_ds = dataset.split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    by_replicate=True  # Ensures no data leakage across replicates
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Access individual samples
for item in train_ds:
    print(f"Image shape: {item.image_array.shape}")  # (H, W, 3) uint8
    print(f"Ground-truth distances: {item.distances.shape}")  # (num_sequences,)
```

#### Complete Workflow Example

```bash
#!/bin/bash

# Generate 10 replicates with 20 sequences each
biomodelml-generate-sequences synth_data/ \
  --preset training \
  --num-replicates 10 \
  --num-sequences 20 \
  --sequence-type N \
  --seed 12345

# Generate HDF5-backed image shards
biomodelml-generate-images synth_data/sequences/ synth_images/ N \
  --max-window 255 \
  --num-workers 8

# Validate the generated HDF5 shards and checksums
biomodelml-validate-images synth_images/

# Now use in Python for model training
```

```python
from biomodelml.data_generation import TrainingDataset
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = TrainingDataset("synth_images/", lazy_load=True)
train_ds, val_ds, test_ds = dataset.split()

# Create DataLoaders for your model
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Train your model here...
for batch in train_loader:
    images = batch.image_array  # Shape: (batch_size, H, W, 3)
    distances = batch.distances  # Shape: (batch_size, num_sequences)
    # ... training code ...
```

### 3. Downstream Task: Phylogenetic Analysis

BioModelML ships with built-in models specifically designed for sequence comparison and phylogenetic reconstruction using our image matrices.

**Via CLI:**

```bash
# Reconstruct a phylogenetic tree using image similarity algorithms
biomodelml-tree mysequences.fasta.N.sanitized results/ N
```

**Via Python API:**

```python
from biomodelml import Experiment
from biomodelml.variants import DeepSearchVariant, OpticalFlowVariant
from pathlib import Path

# Run phylogenetic analysis using pre-trained deep learning features
experiment = Experiment(
    Path("output/"),
    DeepSearchVariant("sequences.fasta.N.sanitized", seq_type="N")
)
experiment.run_and_save()

# Or use optical flow for alignment-free distance computation
# with advanced noise filtering and performance optimization
experiment_optflow = Experiment(
    Path("output_optflow/"),
    OpticalFlowVariant(
        "sequences.fasta.N.sanitized", "N", "output_tensors/",
        optflow_mode='strict',        # Strict preset: aggressive denoising + diagonal focus + high-pass
        profile='accurate',           # Pyramid profile: 'fast', 'accurate', 'sensitive'
        magnitude_threshold=0.5,      # Aggressive thresholding (recommended 0.5-1.0)
        diagonal_ribbon_width=50,     # Focus on diagonal region where evolutionary signal is strongest
        highpass_enabled=True         # High-pass preprocessing to sharpen diagonal signal
    )
)
experiment_optflow.run_and_save()
```

##  Available Task Models (Phylogeny)

BioModelML currently includes several baseline and advanced algorithms for comparing sequence-images to infer evolutionary distance:

**Deep Learning Features:**

- **Deep Search**: Extracts features using a pre-trained VGG16 network and computes distances via Annoy approximate nearest neighbor search. Excellent for large datasets.

**Novel Computer Vision Methods:**

- **Optical Flow**: Alignment-free distance computation using dense optical flow (Farneback algorithm). Measures structural movement between RGB matrices with biochemical channel weighting. Features include:
  - **Pyramid Profiles**: 'fast' (3 levels), 'accurate' (5 levels), 'sensitive' (7 levels) for different tracking depths
    - **Strict Mode**: `optflow_mode='strict'` enables aggressive denoising defaults for phylogenetic signal extraction
    - **Magnitude Thresholding**: Filters weak motion noise (legacy default: 0.0; strict recommended: 0.5-1.0)
    - **Diagonal Ribbon Masking**: Focuses computation on relevant diagonal regions, ignoring empty corners in large matrices
    - **High-pass Preprocessing**: Optional edge enhancement before flow computation (enabled by default in strict mode)
    - **PNG-only Enforcement**: Optical flow requires PNG matrix inputs to avoid JPEG compression artifacts
  
  Particularly effective for detecting evolutionary changes in sequence patterns without traditional alignment. Optimized for sequences >100 residues with significant divergence.

**CLI Strict Optical Flow Example:**

```bash
biomodelml-tree mysequences.fasta.N.sanitized results/ N \
    --algorithms optflow \
    --optflow-mode strict \
    --optflow-threshold 0.5 \
    --optflow-diagonal-width 50 \
    --optflow-highpass
```

**Image Similarity Baselines (SSIM Family):**

- **RMS-SSIM**: Resized Multi-Scale SSIM (Recommended general-purpose baseline)
- **US-SSIM / GS-SSIM**: Unrestricted and Greedy Sliced SSIM (Optimized for varying sequence lengths)
- **WMS-SSIM**: Windowed Multi-Scale SSIM
- **UQI**: Universal Quality Index

*(Traditional string-alignment methods like Needleman-Wunsch and Smith-Waterman are also included for benchmarking purposes).*

##  Documentation & Support

### CLI Commands

- **Core Sequence Processing:**
  - `biomodelml-sanitize` - Clean and validate FASTA sequences
  - `biomodelml-matchmatrix` - Generate RGB matrices from sequences
  - `biomodelml-tree` - Reconstruct phylogenetic trees using various algorithms

- **Data Generation Pipeline:**
  - `biomodelml-generate-sequences` - Create synthetic evolutionary datasets with AliSim (requires IQ-TREE)
  - `biomodelml-generate-images` - Convert FASTA sequences to HDF5-backed RGB image shards
  - `biomodelml-validate-images` - Audit generated HDF5 shards, manifests, and checksums

**Tip:** Run any command with `--help` to see all available options:
```bash
biomodelml-generate-sequences --help
biomodelml-generate-images --help
```

### Learning Resources

- **Jupyter Notebooks**: Check the `notebooks/` directory for tutorials on:
  - Loading and processing image matrices
  - Integrating BioModelML output with PyTorch DataLoaders
  - Training custom deep learning models with phylogenetic ground truth
  
- **Example Workflows**: The supervised data generation pipeline above demonstrates a complete end-to-end workflow from synthetic sequence generation to model training.

- **API Documentation**: Explore the main `biomodelml.data_generation` module for programmatic access to all pipeline components.

### Troubleshooting

- **IQ-TREE not found**: Make sure IQ-TREE is installed and in your system PATH. Run `iqtree -version` to verify.
- **CUDA/GPU issues**: For GPU acceleration, ensure NVIDIA CUDA toolkit and cuDNN are properly installed.
- **Memory issues with large datasets**: Use the `--num-workers` flag to adjust parallel processing during image generation.

## 🤝 Contributing

Contributions are welcome! If you have developed a new deep learning model that utilizes BioModelML matrices for a novel downstream task, please see `CONTRIBUTING.md` to add it to the framework.

##  License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use BioModelML representations in your research, please cite:

[Citation information to be added]