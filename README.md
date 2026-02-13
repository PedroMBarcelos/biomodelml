# BioModelML

A Python framework for bioinformatics sequence analysis using self-comparison matrices and image-based similarity algorithms for DNA, RNA, and protein sequences.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Multiple Similarity Algorithms**: SSIM-based, alignment-based, and deep learning approaches
- **Phylogenetic Tree Construction**: Automated dendrogram generation from sequence data
- **Image-Based Analysis**: Novel RGB matrix representation of sequence comparisons
- **Easy-to-Use CLI**: Command-line tools for common bioinformatics tasks
- **Python API**: Programmatic access for custom workflows
- **GPU Support**: Optional TensorFlow GPU acceleration

## Installation

```bash
# Basic installation
pip install biomodelml

# With all features (deep learning, optimization, etc.)
pip install biomodelml[full]

# For GPU acceleration
pip install biomodelml[gpu]

# For development
pip install biomodelml[dev]
```

**Optional External Tools:**
- [Clustal Omega](http://www.clustal.org/omega/) for multiple sequence alignment: `conda install -c bioconda clustalo`

## Usage

### Command-Line Interface

BioModelML provides simple commands for common workflows:

**1. Prepare your sequences:**
```bash
biomodelml-sanitize mysequences.fasta N  # N=nucleotides, P=proteins
```

**2. Build phylogenetic trees:**
```bash
biomodelml-tree mysequences.fasta.N.sanitized results/ N
```

**3. Use specific algorithms:**
```bash
# Only SSIM-based algorithms
biomodelml-tree sequences.fasta.N.sanitized results/ N --algorithms rmsssim ussim gssim
```

**Advanced Options:**
```bash
# Generate comparison matrices as images
biomodelml-matchmatrix sequences.fasta.N.sanitized images/ N

# Optimize algorithm hyperparameters
biomodelml-optimize data/ sequence_name --trials 200
```

### Python API

Use BioModelML in your Python code:

```python
from biomodelml import Experiment
from biomodelml.variants import (
    ResizedSSIMMultiScaleVariant,
    UnrestrictedSSIMVariant,
)
from pathlib import Path

# Run analysis with multiple algorithms
experiment = Experiment(
    Path("output/"),
    ResizedSSIMMultiScaleVariant("sequences.fasta.N.sanitized", "N"),
    UnrestrictedSSIMVariant("sequences.fasta.N.sanitized", "N"),
)

experiment.run_and_save()
# Results saved to output/ directory
```

**Generate matrices programmatically:**
```python
from biomodelml.matrices import build_matrix
from Bio.Seq import Seq

seq = Seq("ATCGATCG")
matrix = build_matrix(seq, seq, max_window=255, seq_type="N")
# Returns RGB matrix (height, width, 3)
```


## Available Algorithms

### Image-Based Similarity (SSIM Family)
- **RMS-SSIM** - Resized Multi-Scale SSIM (recommended for most cases)
- **US-SSIM** - Unrestricted Sliced SSIM (diagonal search)
- **GS-SSIM** - Greedy Sliced SSIM (optimized search)
- **WMS-SSIM** - Windowed Multi-Scale SSIM (sliding window)
- **R-SSIM** - Resized SSIM (basic version)
- **UQI** - Universal Quality Index

### Traditional Alignment
- **Smith-Waterman** - Local sequence alignment
- **Needleman-Wunsch** - Global sequence alignment
- **Clustal Omega** - Multiple sequence alignment (requires external tool)

### Deep Learning
- **Deep Search** - VGG16 feature extraction + Annoy nearest neighbor search

## How It Works

BioModelML converts sequences into RGB image matrices for comparison:

**Nucleotide Sequences (DNA/RNA):**
- **Red channel**: Self-comparison (sequence vs itself)
- **Green channel**: Complementary comparison
- **Blue channel**: Non-matching positions

**Protein Sequences:**
- **Red channel**: Substitution matrix scores (ProtSub)
- **Green channel**: Self-comparison  
- **Blue channel**: Sneath similarity scores

These matrices are then compared using image similarity algorithms (SSIM) or deep learning to calculate phylogenetic distances.

## Documentation

- **CLI Help**: Run any command with `--help` flag (e.g., `biomodelml-tree --help`)
- **API Reference**: See docstrings in source code for detailed API documentation
- **Examples**: Check `notebooks/` for Jupyter notebook examples

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=biomodelml --cov-report=html

# Specific test file
pytest tests/test_experiment.py -v
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use BioModelML in your research, please cite:
```
[Citation information to be added]
```

## Support

- **Issues**: [GitHub Issues](https://github.com/BioBD/biomodelml/issues)
- **Documentation**: [GitHub Wiki](https://github.com/BioBD/biomodelml/wiki)
