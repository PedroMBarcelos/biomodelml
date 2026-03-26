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

##  Usage

### 1. The Core Pipeline: Sequence to Matrix

Generate image matrices programmatically to feed directly into your custom PyTorch DataLoader.

```bash
# Sanitize sequences
biomodelml-sanitize mysequences.fasta N

# Generate minimal, full-RGB matrices for an entire dataset
biomodelml-matchmatrix mysequences.fasta.N.sanitized output_tensors/ N
```

### 2. Downstream Task: Phylogenetic Analysis

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

- **CLI Help**: Run any command with the `--help` flag (e.g., `biomodelml-tree --help`).
- **Examples**: Check the `notebooks/` directory for Jupyter tutorials on integrating BioModelML matrices with custom PyTorch training loops.
- **Issues**: [GitHub Issues](https://github.com/yourusername/biomodelml/issues)

## 🤝 Contributing

Contributions are welcome! If you have developed a new deep learning model that utilizes BioModelML matrices for a novel downstream task, please see `CONTRIBUTING.md` to add it to the framework.

##  License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use BioModelML representations in your research, please cite:

[Citation information to be added]