"""
BioModelML data generation and training pipeline.

Provides tools for generating synthetic sequence datasets with AliSim
and converting them to image matrices for deep learning.
"""

from biomodelml.data_generation.presets import (
    PRESETS,
    get_preset,
    AliSimConfig,
)
from biomodelml.data_generation.alisim_generator import (
    AliSimGenerator,
    SequenceGenerationJob,
)
from biomodelml.data_generation.image_generator import (
    ImageGenerator,
    ImageMetadata,
)
from biomodelml.data_generation.training_dataset import (
    TrainingDataset,
    ImageTreePair,
    DatasetSplitter,
)

__all__ = [
    "PRESETS",
    "get_preset",
    "AliSimConfig",
    "AliSimGenerator",
    "SequenceGenerationJob",
    "ImageGenerator",
    "ImageMetadata",
    "TrainingDataset",
    "ImageTreePair",
    "DatasetSplitter",
]
