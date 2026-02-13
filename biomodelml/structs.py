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