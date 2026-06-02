"""
Image generation pipeline for sequence matrices.

Orchestrates batch generation of self-comparison RGB matrices from FASTA files,
with metadata tracking for training dataset lineage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from Bio import SeqIO

from biomodelml.matrices import build_matrix
from biomodelml.data_generation.hdf5_store import HDF5ImageWriter
from biomodelml.structs import ImageMetadata


class ImageGenerator:
    """Generate RGB matrices from FASTA files with metadata tracking."""
    
    def __init__(
        self,
        output_dir: str,
        max_window: int = 255,
        num_workers: int = 4,
        include_metadata: bool = True,
    ):
        """
        Initialize image generator.
        
        Args:
            output_dir: Root output directory
            max_window: Maximum matrix dimension (default: 255)
            num_workers: Number of parallel threads (default: 4)
            include_metadata: Track images in metadata manifest (default: True)
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.max_window = max_window
        self.num_workers = num_workers
        self.include_metadata = include_metadata
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.generated_images: List[ImageMetadata] = []
    
    def generate_from_directory(
        self,
        fasta_dir: str,
        sequence_type: str,
        link_tree_distances: bool = True,
    ) -> None:
        """
        Recursively generate images from all FASTA files in a directory.
        
        Args:
            fasta_dir: Directory containing FASTA files
            sequence_type: 'N' for nucleotide, 'P' for protein
            link_tree_distances: Try to link generated tree distances (default: True)
        """
        fasta_dir = Path(fasta_dir)
        
        # Find all FASTA files
        fasta_files = list(fasta_dir.glob("**/*.fasta")) + list(fasta_dir.glob("**/*.fa"))

        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

        self.generate_from_files(
            fasta_files=fasta_files,
            sequence_type=sequence_type,
            link_tree_distances=link_tree_distances,
        )

    def generate_from_files(
        self,
        fasta_files: List[Path],
        sequence_type: str,
        link_tree_distances: bool = True,
    ) -> None:
        """
        Generate images from an explicit FASTA file list.

        Args:
            fasta_files: List of FASTA file paths to process
            sequence_type: 'N' for nucleotide, 'P' for protein
            link_tree_distances: Try to link generated tree distances (default: True)
        """
        if not fasta_files:
            raise FileNotFoundError("No FASTA files to process")

        print(f"Found {len(fasta_files)} FASTA files. Generating images...")

        # Generate images with parallel processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for fasta_file in fasta_files:
                future = executor.submit(
                    self._process_fasta_file,
                    fasta_file,
                    sequence_type,
                    link_tree_distances,
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    self.generated_images.extend(result)
                    print(f"✓ Processed {i+1}/{len(futures)} files")
                except Exception as e:
                    print(f"✗ Error processing file {i+1}: {e}")

        if not self.generated_images:
            raise RuntimeError("No images were generated; check the input files and sequence type")
        
        # Write manifest
        self._write_manifest()
    
    def _process_fasta_file(
        self,
        fasta_file: Path,
        sequence_type: str,
        link_tree_distances: bool,
    ) -> List[ImageMetadata]:
        """
        Process a single FASTA file and generate images for all sequences.
        
        Args:
            fasta_file: Path to FASTA file
            sequence_type: 'N' or 'P'
            link_tree_distances: Try to find corresponding tree distances file
            
        Returns:
            List of ImageMetadata for generated images
        """
        images = []

        shard_path = self.images_dir / f"{fasta_file.stem}.h5"

        # Read sequences
        records = list(SeqIO.parse(fasta_file, "fasta"))

        with HDF5ImageWriter(
            file_path=shard_path,
            source_fasta=str(fasta_file),
            sequence_type=sequence_type,
            max_window=self.max_window,
            num_workers=self.num_workers,
            include_metadata=self.include_metadata,
        ) as writer:
            # Generate self-comparison images
            for record in records:
                # Generate image ID from FASTA name and sequence name
                image_id = f"{fasta_file.stem}_{record.id}"

                # Generate matrix (self-comparison)
                try:
                    matrix = build_matrix(record.seq, record.seq, self.max_window, sequence_type)
                except Exception as e:
                    print(f"Warning: Could not generate matrix for {image_id}: {e}")
                    continue

                # Try to find tree distances file
                distances_path = None
                if link_tree_distances:
                    possible_distances = fasta_file.parent.parent / "trees" / fasta_file.parent.name / f"{fasta_file.stem}.distances.csv"
                    if possible_distances.exists():
                        distances_path = str(possible_distances)

                metadata = writer.write_matrix(
                    image_id=image_id,
                    matrix=matrix,
                    source_fasta=str(fasta_file),
                    sequence_name=record.id,
                    sequence_length=len(record.seq),
                    tree_distances_path=distances_path,
                )

                images.append(metadata)

        if not images:
            raise ValueError(f"No valid sequences could be converted for {fasta_file}")

        return images
    
    def _write_manifest(self) -> None:
        """Write image manifest JSON file."""
        manifest = {
            "generation_date": datetime.now().isoformat(),
            "parameters": {
                "max_window": self.max_window,
                "num_workers": self.num_workers,
                "num_images": len(self.generated_images),
                "storage_format": "hdf5",
            },
            "images": [img.__dict__ for img in self.generated_images],
        }
        
        manifest_file = self.metadata_dir / "image_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Generated {len(self.generated_images)} images")
        print(f"✓ Manifest saved to {manifest_file}")
