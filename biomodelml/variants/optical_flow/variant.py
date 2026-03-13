"""
Optical Flow Variant for Phylogenetic Reconstruction.

Main variant class implementing alignment-free phylogenetic distance
computation using dense optical flow on sequence self-comparison matrices.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool
from typing import Tuple, Optional

from biomodelml.variants.variant import Variant
from biomodelml.structs import DistanceStruct
from biomodelml.variants.optical_flow.flow_computer import FlowComputer
from biomodelml.variants.optical_flow.aggregator import FlowAggregator


class OpticalFlowVariant(Variant):
    """
    Phylogenetic reconstruction using dense optical flow.
    
    Quantifies evolutionary distance by measuring the "structural movement"
    required to transform one sequence's RGB self-comparison matrix into another.
    
    Key features:
    - Zero-padding normalization (default) or resize normalization
    - Multi-channel weighted flow (biochemical importance)
    - Masking to exclude padded regions
    - Bidirectional flow for symmetric distances
    """
    
    name = "Dense Optical Flow with Farneback"
    _MODES = {"legacy", "strict"}
    
    def __init__(self, fasta_file: str = None, sequence_type: str = None,
                 image_folder: str = "", normalization_mode: str = "padding",
                 profile: str = None, magnitude_threshold: Optional[float] = None,
                 diagonal_ribbon_width: Optional[int] = None,
                 optflow_mode: str = "legacy", **flow_params):
        """
        Initialize Optical Flow variant.
        
        Args:
            fasta_file: Path to FASTA file with sequences
            sequence_type: "P" for protein, "N" for nucleotide
            image_folder: Directory containing sequence self-comparison PNGs
            normalization_mode: "padding" (zero-pad to L_max) or "resize" (scale to L_max)
            profile: Pyramid profile ('fast', 'accurate', 'sensitive'). 
                    None = use manual parameters or defaults. Recommended: 'accurate'
            magnitude_threshold: Noise filtering threshold (0-1). Higher = more filtering.
                               Legacy default 0.0 (no filtering).
            diagonal_ribbon_width: Width of diagonal ribbon to focus on (pixels).
                                  None = full matrix.
            optflow_mode: 'legacy' keeps historical behavior; 'strict' applies
                         aggressive thresholding, diagonal masking, and high-pass
            **flow_params: Optional Farneback parameters (overrides profile if specified)
        """
        super().__init__(fasta_file, sequence_type)
        self._image_folder = image_folder or ""
        self._normalization_mode = normalization_mode
        if optflow_mode not in self._MODES:
            raise ValueError(f"Unknown optflow_mode '{optflow_mode}'. Choose from: {sorted(self._MODES)}")
        self._optflow_mode = optflow_mode

        if self._optflow_mode == "strict":
            if profile is None:
                profile = "accurate"
            if magnitude_threshold is None:
                magnitude_threshold = 0.5
            if diagonal_ribbon_width is None:
                diagonal_ribbon_width = 50
            flow_params.setdefault('highpass_enabled', True)
        else:
            if magnitude_threshold is None:
                magnitude_threshold = 0.0
            flow_params.setdefault('highpass_enabled', False)

        self._magnitude_threshold = magnitude_threshold
        self._diagonal_ribbon_width = diagonal_ribbon_width
        
        # Initialize flow computer with profile support
        if profile is not None and 'profile' not in flow_params:
            flow_params['profile'] = profile
        self._flow_computer = FlowComputer(**flow_params)
        self._flow_aggregator = FlowAggregator()
    
    def _read_image(self, img_name: str) -> np.ndarray:
        """
        Load RGB image from disk.
        
        Args:
            img_name: Image filename (without path)
            
        Returns:
            RGB numpy array (H, W, 3)
            
        Raises:
            IOError: If image cannot be loaded
        """
        # Search recursively for the image (handles full/ subdirectory structure)
        if not img_name.lower().endswith('.png'):
            raise ValueError(
                f"OpticalFlowVariant only supports PNG matrix images to avoid compression artifacts. Received: {img_name}"
            )

        pattern = os.path.join(self._image_folder, "**", img_name)
        matches = list(glob.iglob(pattern, recursive=True))
        
        if not matches:
            raise IOError(f"Could not find image: {img_name} in {self._image_folder}")
        
        img_path = matches[0]
        if not img_path.lower().endswith('.png'):
            raise ValueError(
                f"OpticalFlowVariant only supports PNG matrix images. Found: {img_path}"
            )

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def _create_diagonal_mask(self, img_shape: tuple, ribbon_width: int) -> np.ndarray:
        """
        Create mask focusing on diagonal ribbon of the self-comparison matrix.
        
        In self-comparison matrices, the diagonal and nearby regions contain
        the most relevant phylogenetic signal. Corners are often empty space
        that adds noise to distance calculations.
        
        Args:
            img_shape: Shape of image (H, W, C)
            ribbon_width: Width of ribbon around diagonal (pixels)
        
        Returns:
            Binary mask (H, W) where 1 = include, 0 = exclude
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create ribbon around main diagonal
        for i in range(h):
            for j in range(w):
                if abs(i - j) <= ribbon_width:
                    mask[i, j] = 1.0
        
        return mask
    
    def _normalize_sizes_with_padding(self, img1: np.ndarray, 
                                      img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                                 np.ndarray, np.ndarray]:
        """
        Normalize image sizes using zero-padding (spec-compliant method).
        
        Pads smaller images with black pixels to match the largest dimension.
        Creates binary masks to identify valid (non-padded) regions.
        
        Args:
            img1: First RGB image (H1, W1, 3)
            img2: Second RGB image (H2, W2, 3)
            
        Returns:
            Tuple of (padded_img1, padded_img2, mask1, mask2)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find maximum dimensions (L_max)
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        # Create black canvases
        padded_img1 = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        padded_img2 = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        
        # Top-left alignment (starting at 0,0)
        padded_img1[:h1, :w1, :] = img1
        padded_img2[:h2, :w2, :] = img2
        
        # Create masks for valid regions
        mask1 = self._flow_aggregator.create_mask(h1, w1, max_h, max_w)
        mask2 = self._flow_aggregator.create_mask(h2, w2, max_h, max_w)
        
        return padded_img1, padded_img2, mask1, mask2
    
    def _normalize_sizes_with_resize(self, img1: np.ndarray, 
                                     img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
        """
        Normalize image sizes using NEAREST interpolation resize.
        
        Resizes both images to match the largest dimension while preserving
        discrete pixel values (important for categorical sequence encodings).
        
        Args:
            img1: First RGB image (H1, W1, 3)
            img2: Second RGB image (H2, W2, 3)
            
        Returns:
            Tuple of (resized_img1, resized_img2, mask1, mask2)
            Note: masks are all ones (no padding to filter)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find maximum dimensions
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        # Resize using NEAREST to preserve categorical encoding
        resized_img1 = cv2.resize(img1, (max_w, max_h), 
                                  interpolation=cv2.INTER_NEAREST)
        resized_img2 = cv2.resize(img2, (max_w, max_h),
                                  interpolation=cv2.INTER_NEAREST)
        
        # Full masks (no padding to filter out)
        mask1 = np.ones((max_h, max_w), dtype=np.float32)
        mask2 = np.ones((max_h, max_w), dtype=np.float32)
        
        return resized_img1, resized_img2, mask1, mask2
    
    def calc_alg(self, img_name1: str, img_name2: str) -> float:
        """
        Compute optical flow distance between two sequence images.
        
        Core pairwise comparison:
        1. Load RGB images
        2. Normalize sizes (padding or resize)
        3. Compute multi-channel bidirectional flow
        4. Apply biochemical weights
        5. Aggregate masked distance
        
        Args:
            img_name1: First image filename
            img_name2: Second image filename
            
        Returns:
            Phylogenetic distance (higher = more different)
        """
        # Load images
        img1 = self._read_image(img_name1)
        img2 = self._read_image(img_name2)
        
        # Normalize sizes based on mode
        if self._normalization_mode == "padding":
            norm_img1, norm_img2, mask1, mask2 = self._normalize_sizes_with_padding(img1, img2)
        elif self._normalization_mode == "resize":
            norm_img1, norm_img2, mask1, mask2 = self._normalize_sizes_with_resize(img1, img2)
        else:
            raise ValueError(f"Unknown normalization mode: {self._normalization_mode}")
        
        # Apply diagonal ribbon mask if specified
        if self._diagonal_ribbon_width is not None:
            diagonal_mask = self._create_diagonal_mask(
                norm_img1.shape, 
                self._diagonal_ribbon_width
            )
            # Combine with padding masks
            mask1 = mask1 * diagonal_mask
            mask2 = mask2 * diagonal_mask
        
        # Compute bidirectional multi-channel flow
        forward_flows, backward_flows = self._flow_computer.bidirectional_flow(
            norm_img1, norm_img2
        )
        
        # Aggregate distance with weighting, masking, and thresholding
        distance = self._flow_aggregator.compute_bidirectional_distance(
            forward_flows, backward_flows,
            mask1, mask2,
            self._sequence_type,
            magnitude_threshold=self._magnitude_threshold
        )
        
        # Extract sequence names for logging
        seq1 = ".".join(img_name1.split('.')[:-1])
        seq2 = ".".join(img_name2.split('.')[:-1])
        print(f"{seq1} and {seq2} done with distance {distance:.4f}")
        
        return distance
    
    def build_matrix(self) -> DistanceStruct:
        """
        Build pairwise distance matrix using optical flow.
        
        Parallelizes comparisons using multiprocessing for efficiency.
        
        Returns:
            DistanceStruct containing sequence names and symmetric distance matrix
            
        Raises:
            IOError: If image files for sequences are missing
        """
        # Ensure image folder is set
        if not self._image_folder:
            raise ValueError("image_folder must be specified for OpticalFlowVariant")
        
        # Reject compressed image formats to avoid optical-flow artifacts
        non_png_patterns = ["**/*.jpg", "**/*.jpeg", "**/*.JPG", "**/*.JPEG"]
        non_png_files = []
        for non_png_pattern in non_png_patterns:
            non_png_files.extend(glob.glob(os.path.join(self._image_folder, non_png_pattern), recursive=True))
        if non_png_files:
            sample = ", ".join(non_png_files[:3])
            raise IOError(
                "OpticalFlowVariant requires PNG-only matrix inputs. "
                f"Found compressed files: {sample}"
            )

        # Discover available image files (search recursively)
        pattern = os.path.join(self._image_folder, "**", "*.png")
        all_images = list(glob.iglob(pattern, recursive=True))
        
        # Extract basenames without extension
        indexes = {}
        for img_path in all_images:
            basename = os.path.basename(img_path)
            name = ".".join(basename.split('.')[:-1])
            ext = basename.split('.')[-1]
            indexes[name] = ext
        
        # Validate all sequences have corresponding images
        diff = set(self._names).difference(set(indexes.keys()))
        if diff:
            raise IOError(f"Sequences without image created: {diff}")
        
        # Build full image filenames
        files = [f"{name}.{indexes[name]}" for name in self._names]
        
        # Initialize distance matrix
        df = pd.DataFrame(index=self._names, columns=self._names)
        last_ids = []
        threads = 3  # Optimal for I/O-bound optical flow computation
        
        print(f"\nComputing optical flow distances for {len(files)} sequences")
        print(f"Normalization mode: {self._normalization_mode}")
        print(f"Sequence type: {self._sequence_type}")
        
        # Compute upper triangle of symmetric matrix with parallelization
        for idx, img1 in enumerate(files):
            idx1 = self._names[idx]
            
            # Parallelize pairwise comparisons for this row
            with Pool(threads) as pool:
                result = pool.starmap(
                    self.calc_alg,
                    [(img1, img2) for img2 in files[idx:]]
                )
            
            # Fill symmetric distance matrix
            if last_ids:
                df.loc[idx1, self._names[idx:]] = result
                df.loc[idx1, last_ids] = df.loc[last_ids, idx1]
            else:
                df.loc[idx1, :] = result
            last_ids.append(idx1)
        
        print(f"\n✓ Optical flow distance matrix complete")
        
        return DistanceStruct(
            names=self._names,
            matrix=df.to_numpy(np.float64)
        )
