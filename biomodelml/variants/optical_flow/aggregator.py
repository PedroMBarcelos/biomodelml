"""
Distance Aggregation from Optical Flow Fields.

Aggregates flow magnitudes into phylogenetic distance metrics with support
for masking (to exclude padding regions) and biochemical weighting of RGB channels.
"""

import numpy as np
from typing import Dict, Tuple


class FlowAggregator:
    """
    Converts optical flow fields into phylogenetic distance metrics.
    
    Handles:
    - Binary masking to exclude padded regions
    - Biochemical weighting of RGB channels based on sequence encoding
    - Symmetric distance computation from bidirectional flows
    """
    
    @staticmethod
    def create_mask(original_h: int, original_w: int, 
                   padded_h: int, padded_w: int) -> np.ndarray:
        """
        Create binary mask marking valid sequence region vs padding.
        
        The mask is 1 (True) for pixels representing actual sequence
        comparisons and 0 (False) for padded black pixels.
        
        Args:
            original_h: Original image height before padding
            original_w: Original image width before padding
            padded_h: Padded image height
            padded_w: Padded image width
            
        Returns:
            Binary mask (padded_h, padded_w) with 1s in valid region
        """
        mask = np.zeros((padded_h, padded_w), dtype=np.float32)
        mask[:original_h, :original_w] = 1.0
        return mask
    
    @staticmethod
    def get_rgb_weights(sequence_type: str) -> Dict[str, float]:
        """
        Get biochemical weights for RGB channels based on sequence type.
        
        These weights reflect the biological importance of each channel:
        
        Protein sequences (P):
        - Red (BLOSUM substitution scores): Most important (1.0)
        - Green (identity): Medium importance (0.6)
        - Blue (Sneath hydrophobic similarity): Lower importance (0.3)
        
        Nucleotide sequences (N):
        - Red (direct matches): Most important (1.0)
        - Green (complementary pairing): High importance (0.8)
        - Blue (non-matches): Lower importance (0.2)
        
        Args:
            sequence_type: "P" for protein, "N" for nucleotide
            
        Returns:
            Dictionary mapping 'red', 'green', 'blue' to weight values
        """
        if sequence_type == "P":
            return {'red': 1.0, 'green': 0.6, 'blue': 0.3}
        elif sequence_type == "N":
            return {'red': 1.0, 'green': 0.8, 'blue': 0.2}
        else:
            # Default: equal weights
            return {'red': 1.0, 'green': 1.0, 'blue': 1.0}
    
    @staticmethod
    def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
        """
        Compute magnitude from flow field vectors.
        
        Args:
            flow: Flow field (H, W, 2) where [:,:,0]=u, [:,:,1]=v
            
        Returns:
            Magnitude array (H, W) with sqrt(u^2 + v^2) at each pixel
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        return magnitude
    
    def aggregate_distance(self, flows: Dict[str, np.ndarray],
                          mask: np.ndarray, sequence_type: str,
                          magnitude_threshold: float = 0.0) -> float:
        """
        Aggregate flow magnitudes into single distance value.
        
        Applies biochemical weighting and masks out padding regions.
        Optionally applies magnitude thresholding to filter background noise.
        
        Formula:
            distance = sum(mask * weighted_magnitude) / sum(mask)
        
        Where weighted_magnitude combines RGB channels with biochemical weights.
        
        Args:
            flows: Dictionary with 'red', 'green', 'blue' flow fields
            mask: Binary mask (1=valid, 0=padding)
            sequence_type: "P" or "N" for weighting scheme
            magnitude_threshold: Minimum magnitude to include (filters noise).
                               Values below threshold are set to 0.
            
        Returns:
            Aggregated distance value (higher = more different)
        """
        weights = self.get_rgb_weights(sequence_type)
        
        # Compute magnitude for each channel
        mag_r = self.compute_flow_magnitude(flows['red'])
        mag_g = self.compute_flow_magnitude(flows['green'])
        mag_b = self.compute_flow_magnitude(flows['blue'])
        
        # Apply magnitude thresholding (denoising)
        if magnitude_threshold > 0:
            mag_r = np.where(mag_r > magnitude_threshold, mag_r, 0.0)
            mag_g = np.where(mag_g > magnitude_threshold, mag_g, 0.0)
            mag_b = np.where(mag_b > magnitude_threshold, mag_b, 0.0)
        
        # Apply biochemical weights
        weighted_magnitude = (
            weights['red'] * mag_r +
            weights['green'] * mag_g +
            weights['blue'] * mag_b
        )
        
        # Apply mask and normalize by valid area
        masked_magnitude = weighted_magnitude * mask
        distance = np.sum(masked_magnitude) / np.sum(mask) if np.sum(mask) > 0 else 0.0
        
        return distance
    
    def compute_bidirectional_distance(self, 
                                      forward_flows: Dict[str, np.ndarray],
                                      backward_flows: Dict[str, np.ndarray],
                                      mask1: np.ndarray, mask2: np.ndarray,
                                      sequence_type: str,
                                      magnitude_threshold: float = 0.0) -> float:
        """
        Compute symmetric distance from bidirectional flows.
        
        Averages forward and backward distances for robustness.
        
        Args:
            forward_flows: Flow from img1 to img2
            backward_flows: Flow from img2 to img1
            mask1: Mask for img1
            mask2: Mask for img2
            sequence_type: "P" or "N"
            magnitude_threshold: Minimum magnitude to include (filters noise)
            
        Returns:
            Averaged bidirectional distance
        """
        dist_forward = self.aggregate_distance(
            forward_flows, mask1, sequence_type, magnitude_threshold
        )
        dist_backward = self.aggregate_distance(
            backward_flows, mask2, sequence_type, magnitude_threshold
        )
        
        # Average for symmetric distance
        return (dist_forward + dist_backward) / 2.0
