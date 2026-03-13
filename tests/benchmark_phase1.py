"""
Performance Benchmarks for Phase 1 Refactoring
Measures improvements in interpolation speed, I/O throughput, and storage efficiency.
"""

import time
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from Bio.Seq import Seq
from Bio import SeqIO
from biomodelml.matrices import build_matrix, save_image_by_matrices


def benchmark_interpolation():
    """Benchmark NEAREST vs CUBIC interpolation performance."""
    print("\n=== Interpolation Performance ===")
    
    # Create test matrix
    seq = Seq("ATGCATGCTAGCATGC" * 10)  # 160 bases
    matrix = build_matrix(seq, seq, 255, "N")
    
    target_sizes = [(64, 64), (128, 128), (256, 256)]
    iterations = 100
    
    for size in target_sizes:
        print(f"\nResize to {size}:")
        
        # Benchmark NEAREST (current)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = cv2.resize(matrix, size, interpolation=cv2.INTER_NEAREST)
        nearest_time = time.perf_counter() - start
        
        # Benchmark CUBIC (old)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = cv2.resize(matrix, size, interpolation=cv2.INTER_CUBIC)
        cubic_time = time.perf_counter() - start
        
        speedup = cubic_time / nearest_time
        print(f"  NEAREST: {nearest_time:.4f}s ({iterations} iterations)")
        print(f"  CUBIC:   {cubic_time:.4f}s ({iterations} iterations)")
        print(f"  Speedup: {speedup:.2f}x")


def benchmark_image_generation():
    """Benchmark full image generation pipeline."""
    print("\n=== Image Generation Throughput ===")
    
    # Load test sequences
    test_data = Path(__file__).parent.parent / "data" / "example_sequences.fasta"
    if not test_data.exists():
        print("Skipping: test data not found")
        return
    
    sequences = list(SeqIO.parse(str(test_data), "fasta"))[:10]  # First 10 sequences
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Benchmark minimal mode
        start = time.perf_counter()
        for seq in sequences:
            save_image_by_matrices(
                seq.id, seq.id, seq.seq, seq.seq,
                255, tmpdir, "N",
                generate_variations=False
            )
        minimal_time = time.perf_counter() - start
        
        # Benchmark extended mode
        tmpdir2 = tempfile.mkdtemp()
        start = time.perf_counter()
        for seq in sequences:
            save_image_by_matrices(
                seq.id, seq.id, seq.seq, seq.seq,
                255, tmpdir2, "N",
                generate_variations=True
            )
        extended_time = time.perf_counter() - start
        shutil.rmtree(tmpdir2)
    
    n_seqs = len(sequences)
    print(f"\nGenerated {n_seqs} self-comparison images:")
    print(f"  Minimal mode:  {minimal_time:.2f}s ({n_seqs/minimal_time:.1f} seq/s)")
    print(f"  Extended mode: {extended_time:.2f}s ({n_seqs/extended_time:.1f} seq/s)")
    print(f"  Speedup: {extended_time/minimal_time:.2f}x")


def benchmark_storage_efficiency():
    """Benchmark storage reduction from minimal mode."""
    print("\n=== Storage Efficiency ===")
    
    seq = Seq("ATGCATGC" * 20)  # 160 bases
    
    with tempfile.TemporaryDirectory() as tmpdir:
        minimal_path = Path(tmpdir) / "minimal"
        extended_path = Path(tmpdir) / "extended"
        minimal_path.mkdir()
        extended_path.mkdir()
        
        # Generate minimal
        save_image_by_matrices(
            "test", "test", seq, seq,
            255, str(minimal_path), "N",
            generate_variations=False
        )
        
        # Generate extended
        save_image_by_matrices(
            "test", "test", seq, seq,
            255, str(extended_path), "N",
            generate_variations=True
        )
        
        # Calculate sizes
        def get_size(path):
            total = 0
            for f in path.rglob("*.png"):
                total += f.stat().st_size
            return total
        
        minimal_size = get_size(minimal_path)
        extended_size = get_size(extended_path)
        
        reduction = (1 - minimal_size / extended_size) * 100
        
        print(f"\nStorage for single sequence:")
        print(f"  Minimal:  {minimal_size:,} bytes (1 image)")
        print(f"  Extended: {extended_size:,} bytes (11 images)")
        print(f"  Reduction: {reduction:.1f}%")


def benchmark_io_methods():
    """Benchmark cv2.imwrite vs matplotlib.pyplot.imsave."""
    print("\n=== I/O Performance ===")
    
    seq = Seq("ATGCATGC" * 20)
    matrix = build_matrix(seq, seq, 255, "N")
    iterations = 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Benchmark cv2.imwrite (current)
        start = time.perf_counter()
        for i in range(iterations):
            # Convert RGB to BGR for cv2
            bgr = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{tmpdir}/cv2_{i}.png", bgr)
        cv2_time = time.perf_counter() - start
        
        # Benchmark matplotlib (old method)
        try:
            import matplotlib.pyplot as plt
            start = time.perf_counter()
            for i in range(iterations):
                plt.imsave(f"{tmpdir}/mpl_{i}.png", matrix)
            mpl_time = time.perf_counter() - start
            
            speedup = mpl_time / cv2_time
            print(f"\nSaving {iterations} images:")
            print(f"  cv2.imwrite:      {cv2_time:.4f}s ({iterations/cv2_time:.1f} img/s)")
            print(f"  matplotlib.imsave: {mpl_time:.4f}s ({iterations/mpl_time:.1f} img/s)")
            print(f"  Speedup: {speedup:.2f}x")
        except ImportError:
            print("\nmatplotlib not installed, skipping comparison")
            print(f"  cv2.imwrite: {cv2_time:.4f}s ({iterations/cv2_time:.1f} img/s)")


def benchmark_pairwise_matrix():
    """Benchmark pairwise comparison performance."""
    print("\n=== Pairwise Comparison Performance ===")
    
    # Create two sequences with varying similarity
    seq1 = Seq("ATGCATGC" * 20)
    seq2 = Seq("ATGCTTGC" * 20)  # 1 mismatch per 8 bases
    
    iterations = 100
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = build_matrix(seq1, seq2, 255, "N")
    elapsed = time.perf_counter() - start
    
    print(f"\nPairwise matrix generation ({iterations} iterations):")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {iterations/elapsed:.1f} matrices/s")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Performance Benchmarks")
    print("=" * 60)
    
    benchmark_interpolation()
    benchmark_io_methods()
    benchmark_image_generation()
    benchmark_storage_efficiency()
    benchmark_pairwise_matrix()
    
    print("\n" + "=" * 60)
    print("Benchmarks Complete")
    print("=" * 60)
