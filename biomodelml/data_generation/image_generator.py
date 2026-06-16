"""
Image generation pipeline for sequence matrices.

Orchestrates batch generation of self-comparison RGB matrices from FASTA files,
with metadata tracking for training dataset lineage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List
#from concurrent.futures import ThreadPoolExecutor ProcessesPoolExecutor
from concurrent.futures import ProcessPoolExecutor
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
        input_dir: Path = None
    ) -> None:
        """
        Generate images from an explicit FASTA file list.

        Args:
            fasta_files: List of FASTA file paths to process
            sequence_type: 'N' for nucleotide, 'P' for protein
            link_tree_distances: Try to link generated tree distances (default: True)
            input_dir: The base sequence directory (e.g., output_dir/sequences)
        """
        if not fasta_files:
            raise FileNotFoundError("No FASTA files to process")

        print(f"Found {len(fasta_files)} FASTA files. Generating images...")

        # Generate images with parallel processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                self._process_fasta_file, # Note: if using ProcessPool, ensure this method is picklable
                fasta_file=f, 
                sequence_type=sequence_type, 
                link_tree_distances=link_tree_distances,
                input_dir=input_dir
            )
            for f in fasta_files
    ]
            
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
        input_dir: Path = None,
    ) -> List[ImageMetadata]:
        """
        Process a single FASTA file and generate images for all sequences.
        """
        images = []

        # --- Construct a collision-free shard path ---
        if input_dir:
            relative_path = fasta_file.relative_to(input_dir)
            flattened_folder = str(relative_path.parent).replace('/', '_')
            if flattened_folder and flattened_folder != ".":
                shard_name = f"{flattened_folder}_{fasta_file.stem}.h5"
            else:
                shard_name = f"{fasta_file.stem}.h5"
        else:
            parent_name = fasta_file.parent.name
            if parent_name and parent_name != "sequences" and parent_name != ".":
                shard_name = f"{parent_name}_{fasta_file.stem}.h5"
            else:
                shard_name = f"{fasta_file.stem}.h5"

        shard_path = self.images_dir / shard_name

        # Read sequences
        records = list(SeqIO.parse(fasta_file, "fasta"))

        # --- FIX: Find the maximum length in this specific file to tell the writer ---
        # This keeps max_window an integer, preventing the dtype('O') error,
        # but scales it perfectly to accommodate your largest sequence.
        max_len_in_file = max(len(record.seq) for record in records) if records else 255

        with HDF5ImageWriter(
            file_path=shard_path,
            source_fasta=str(fasta_file),
            sequence_type=sequence_type,
            max_window=max_len_in_file,  # ✔️ Passed as an integer now!
            num_workers=self.num_workers,
            include_metadata=self.include_metadata,
        ) as writer:
            
            # Generate self-comparison images
            for record in records:
                image_id = f"{shard_path.stem}_{record.id}"
                current_sequence_len = len(record.seq)
                
                # Generate matrix matching the exact sequence size
                try:
                    matrix = build_matrix(record.seq, record.seq, current_sequence_len, sequence_type)
                except Exception as e:
                    print(f"Warning: Could not generate matrix for {image_id}: {e}")
                    continue

                # Try to find tree distances file
                distances_path = None
                if link_tree_distances:
                    # 1. Pegamos o stem limpo do alinhamento original (ex: 'alignment_001')
                    # Removendo quaisquer sufixos de sanitização ou tipo (.P, .N, .sanitized)
                    clean_stem = fasta_file.name.split('.')[0]
                    
                    # 2. Localizamos a pasta de árvores correspondente à réplica atual
                    # test_dir/sequences/replicate_001 -> test_dir/trees/replicate_001
                    replicate_folder = fasta_file.parent.name
                    sequences_root = fasta_file.parent.parent
                    project_root = sequences_root.parent
                    
                    # Montamos o caminho esperado para o CSV de distâncias
                    possible_distances = project_root / "trees" / replicate_folder / f"{clean_stem}.distances.csv"

                    # Se não achar, tentamos um fallback flexível olhando a pasta da réplica
                    if not possible_distances.exists():
                        trees_dir = project_root / "trees" / replicate_folder
                        if trees_dir.exists():
                            # Procura qualquer CSV que comece com o nome do alinhamento
                            matched_csvs = list(trees_dir.glob(f"{clean_stem}*.csv"))
                            if matched_csvs:
                                possible_distances = matched_csvs[0]

                    if possible_distances.exists():
                        distances_path = str(possible_distances)
                    else:
                        print(f"Warning: Tree distances not found for {image_id} at {possible_distances}")
                # --- FIM DA CORREÇÃO ---

                metadata = writer.write_matrix(
                    image_id=image_id,
                    matrix=matrix,
                    source_fasta=str(fasta_file),
                    sequence_name=record.id,
                    sequence_length=current_sequence_len,
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
