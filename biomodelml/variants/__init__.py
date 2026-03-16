"""
BioModelML similarity algorithm variants.

This module contains all sequence similarity algorithms implemented in BioModelML,
including traditional alignment methods (Smith-Waterman, Needleman-Wunsch) and
novel image-based methods using SSIM and deep learning.
"""

from biomodelml.variants.variant import Variant
from biomodelml.variants.control import ControlVariant
from biomodelml.variants.siamese import SiameseVariant
from biomodelml.variants.sw import SmithWatermanVariant
from biomodelml.variants.nw import NeedlemanWunschVariant
from biomodelml.variants.ssim_base import SSIMVariant
from biomodelml.variants.ssim_multiscale_base import SSIMMultiScaleVariant
from biomodelml.variants.resized_ssim import ResizedSSIMVariant
from biomodelml.variants.resized_ssim_multiscale import ResizedSSIMMultiScaleVariant
from biomodelml.variants.windowed_ssim_multiscale import WindowedSSIMMultiScaleVariant
from biomodelml.variants.greedy_ssim import GreedySSIMVariant
from biomodelml.variants.unrestricted_ssim import UnrestrictedSSIMVariant
from biomodelml.variants.uqi import UQIVariant
from biomodelml.variants.deep_search.variant import DeepSearchVariant
from biomodelml.variants.optical_flow import OpticalFlowVariant

__all__ = [
    "Variant",
    "ControlVariant",
    "SmithWatermanVariant",
    "NeedlemanWunschVariant",
    "SSIMVariant",
    "SSIMMultiScaleVariant",
    "ResizedSSIMVariant",
    "ResizedSSIMMultiScaleVariant",
    "WindowedSSIMMultiScaleVariant",
    "GreedySSIMVariant",
    "UnrestrictedSSIMVariant",
    "UQIVariant",
    "DeepSearchVariant",
    "OpticalFlowVariant",
]
