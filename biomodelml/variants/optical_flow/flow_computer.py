"""
Optical Flow Computation using Farneback Algorithm.

Handles dense optical flow calculation for multi-channel (RGB) images,
with bidirectional flow computation for symmetric distance metrics.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional

# Pyramid profiles optimized for biological sequence analysis
PYRAMID_PROFILES = {
    'fast': {
        'levels': 3,
        'pyr_scale': 0.5,
        'winsize': 15,
        'description': 'Fast processing, suitable for similar sequences'
    },
    'accurate': {
        'levels': 5,
        'pyr_scale': 0.5,
        'winsize': 15,
        'description': 'Better tracking of large-scale movements (recommended)'
    },
    'sensitive': {
        'levels': 7,
        'pyr_scale': 0.6,
        'winsize': 21,
        'description': 'Maximum sensitivity for highly divergent sequences'
    }
}


class FlowComputer:
    """
    Computes dense optical flow using Gunnar Farneback's algorithm.
    
    Supports multi-channel (RGB) flow computation where each color channel
    represents different biochemical properties in sequence comparison matrices.
    """
    
    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2,
                 flags: int = 0, profile: Optional[str] = None,
                 highpass_enabled: bool = False,
                 highpass_kernel_size: int = 3,
                 highpass_strength: float = 1.0):
        """
        Initialize Farneback optical flow parameters.
        
        Args:
            pyr_scale: Pyramid scale factor (0.5 means half size at each level)
            levels: Number of pyramid levels
            winsize: Averaging window size (15-25 recommended for sequence data)
            iterations: Number of iterations at each pyramid level
            poly_n: Size of pixel neighborhood for polynomial expansion
            poly_sigma: Standard deviation of Gaussian for polynomial expansion
            flags: Operation flags (0 for default)
            profile: Preset profile ('fast', 'accurate', 'sensitive'). 
                    If provided, overrides pyr_scale, levels, and winsize.
                highpass_enabled: Apply high-pass preprocessing before flow computation
                highpass_kernel_size: Gaussian kernel size used in high-pass filter (odd >= 3)
                highpass_strength: High-pass boost strength (higher = stronger edge emphasis)
        """
        # Apply profile if specified
        if profile is not None:
            if profile not in PYRAMID_PROFILES:
                raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(PYRAMID_PROFILES.keys())}")
            
            profile_params = PYRAMID_PROFILES[profile]
            pyr_scale = profile_params['pyr_scale']
            levels = profile_params['levels']
            winsize = profile_params['winsize']
        
        if highpass_kernel_size < 3 or highpass_kernel_size % 2 == 0:
            raise ValueError("highpass_kernel_size must be an odd integer >= 3")

        self.params = {
            'pyr_scale': pyr_scale,
            'levels': levels,
            'winsize': winsize,
            'iterations': iterations,
            'poly_n': poly_n,
            'poly_sigma': poly_sigma,
            'flags': flags
        }
        self.profile = profile
        self.highpass_enabled = highpass_enabled
        self.highpass_kernel_size = highpass_kernel_size
        self.highpass_strength = highpass_strength

    def _apply_highpass(self, channel: np.ndarray) -> np.ndarray:
        """
        Apply high-pass enhancement using unsharp masking.

        Args:
            channel: Single-channel normalized image in [0, 1]

        Returns:
            Enhanced channel in [0, 1]
        """
        if not self.highpass_enabled:
            return channel

        blurred = cv2.GaussianBlur(
            channel,
            (self.highpass_kernel_size, self.highpass_kernel_size),
            sigmaX=0
        )
        detail = channel - blurred
        enhanced = channel + (self.highpass_strength * detail)
        return np.clip(enhanced, 0.0, 1.0).astype(np.float32)
    
    def compute_flow_single_channel(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow for a single grayscale channel.
        
        Args:
            gray1: First grayscale image (H, W), uint8 or float32
            gray2: Second grayscale image (H, W), uint8 or float32
            
        Returns:
            Flow field (H, W, 2) where [:,:,0] is horizontal, [:,:,1] is vertical
        """
        # Convert to float32 [0, 1] if needed (Farneback works better with normalized inputs)
        if gray1.dtype == np.uint8:
            gray1 = gray1.astype(np.float32) / 255.0
        if gray2.dtype == np.uint8:
            gray2 = gray2.astype(np.float32) / 255.0

        gray1 = self._apply_highpass(gray1)
        gray2 = self._apply_highpass(gray2)
            
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            **self.params
        )
        return flow
    
    def compute_flow_per_channel(self, img1_rgb: np.ndarray, 
                                  img2_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute optical flow separately for each RGB channel.
        
        This allows biochemical weighting where different channels represent
        different sequence properties:
        - Red: Substitution scores (BLOSUM) or direct matches
        - Green: Identity or complementary base pairing
        - Blue: Hydrophobic similarity (Sneath) or mismatches
        
        Args:
            img1_rgb: First RGB image (H, W, 3)
            img2_rgb: Second RGB image (H, W, 3)
            
        Returns:
            Dictionary with keys 'red', 'green', 'blue' containing flow fields
        """
        flows = {}
        
        # Extract and compute flow for each channel
        flows['red'] = self.compute_flow_single_channel(
            img1_rgb[:, :, 0], img2_rgb[:, :, 0]
        )
        flows['green'] = self.compute_flow_single_channel(
            img1_rgb[:, :, 1], img2_rgb[:, :, 1]
        )
        flows['blue'] = self.compute_flow_single_channel(
            img1_rgb[:, :, 2], img2_rgb[:, :, 2]
        )
        
        return flows
    
    def bidirectional_flow(self, img1: np.ndarray, 
                          img2: np.ndarray) -> Tuple[Dict[str, np.ndarray], 
                                                      Dict[str, np.ndarray]]:
        """
        Compute optical flow in both directions for symmetric distance metric.
        
        Forward flow (img1 -> img2) may differ from backward flow (img2 -> img1).
        Computing both and averaging produces a more robust distance measure.
        
        Args:
            img1: First RGB image (H, W, 3)
            img2: Second RGB image (H, W, 3)
            
        Returns:
            Tuple of (forward_flows, backward_flows) dictionaries
        """
        forward_flows = self.compute_flow_per_channel(img1, img2)
        backward_flows = self.compute_flow_per_channel(img2, img1)
        
        return forward_flows, backward_flows
