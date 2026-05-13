import numpy
from dataclasses import dataclass
from typing import List, Optional, Any
from biotite.sequence.phylo import Tree
from biotite.sequence.align import Alignment


@dataclass
class ImgDebug:
    score: str
    start_col: str
    start_line: str
    stop_col: str
    stop_line: str
    max_size: str

@dataclass
class ImgMap:
    debugs: List[ImgDebug]
    scores: List[float]

@dataclass
class ImgDebugs:
    img1: str
    img2: str
    debugs: List[ImgDebug]

@dataclass
class DistanceStruct:
    names: List[str]
    matrix: numpy.ndarray
    align: Optional[Alignment] = None
    img_debugs: Optional[List[ImgDebugs]] = None


@dataclass
class TreeStruct:
    name: str
    distances: DistanceStruct
    tree: Tree


@dataclass
class SeqTypeStruct:
    N: List[str]
    P: List[str]


@dataclass
class ImageMetadata:
    """Metadata for a single generated image."""
    image_id: str
    image_path: str
    source_fasta: str
    sequence_name: str
    sequence_length: int
    matrix_shape: tuple  # (height, width, channels)
    tree_distances_path: Optional[str] = None
    tree_distances_available: bool = False
    additional_info: Optional[dict] = None


@dataclass
class SequenceGenerationJob:
    """Metadata for a sequence generation run via AliSim."""
    job_id: str
    preset: str
    sequence_id: str
    fasta_path: str
    tree_path: str
    distances_path: str
    num_sequences: int
    alignment_length: int
    sequence_type: str
    random_seed: Optional[int] = None
    parameters: Optional[dict] = None
    timestamp: Optional[str] = None