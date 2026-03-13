"""
Optical Flow-based phylogenetic distance computation.

Uses dense optical flow (Farneback method) to quantify structural differences
between self-comparison sequence images.
"""

from biomodelml.variants.optical_flow.variant import OpticalFlowVariant

__all__ = ["OpticalFlowVariant"]
