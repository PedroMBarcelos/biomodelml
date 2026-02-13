"""
BioModelML - Bioinformatics framework for sequence analysis.

This package provides tools for analyzing DNA, RNA, and protein sequences using
self-comparison matrices and image-based similarity algorithms.
"""

from biomodelml.__version__ import __version__
from biomodelml.experiment import Experiment
from biomodelml.structs import DistanceStruct, TreeStruct
from biomodelml.matrices import build_matrix, save_image_by_matrices
from biomodelml.sanitize import convert_and_remove_unrelated_sequences

# Import variants subpackage
from biomodelml import variants

__all__ = [
    "__version__",
    "Experiment",
    "DistanceStruct",
    "TreeStruct",
    "build_matrix",
    "save_image_by_matrices",
    "convert_and_remove_unrelated_sequences",
    "variants",
]
