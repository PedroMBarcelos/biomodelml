#!/usr/bin/env python3
"""
Diagnose why biomodelml struggles with indels
"""

from pathlib import Path
from Bio import SeqIO
from biomodelml.matrices import build_matrix, save_image_by_matrices
from Bio.Seq import Seq
import numpy as np

def analyze_sequence_lengths(fasta_path: Path):
    """Check if sequences have variable lengths"""
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    
    lengths = [len(s.seq) for s in sequences]
    
    print(f"\n{'='*60}")
    print("SEQUENCE LENGTH ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Length variation: {max(lengths) - min(lengths)} bp")
    print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
    
    print(f"\nIndividual lengths:")
    for s in sequences:
        print(f"  {s.id}: {len(s.seq)} bp")
    
    return sequences, lengths


def test_matrix_generation(sequences):
    """Test what happens when we generate matrices for different length sequences"""
    
    print(f"\n{'='*60}")
    print("MATRIX DIMENSION ANALYSIS")
    print(f"{'='*60}")
    
    seq_type = 'N'  # or 'P' for protein
    
    # Test self-comparison
    print("\n1. Self-comparisons (should be square):")
    for seq_record in sequences[:3]:  # Test first 3
        seq = seq_record.seq
        matrix = build_matrix(seq, seq, 255, seq_type)
        print(f"  {seq_record.id}: {matrix.shape} (length={len(seq)})")
    
    # Test pairwise comparisons
    print("\n2. Pairwise comparisons (may be rectangular):")
    for i in range(min(3, len(sequences))):
        for j in range(i+1, min(3, len(sequences))):
            seq1 = sequences[i].seq
            seq2 = sequences[j].seq
            matrix = build_matrix(seq1, seq2, 255, seq_type)
            print(f"  {sequences[i].id} x {sequences[j].id}: {matrix.shape}")
            print(f"    (lengths: {len(seq1)} x {len(seq2)})")
            
            if matrix.shape[0] != matrix.shape[1]:
                print(f"    ⚠️  NON-SQUARE MATRIX! SSIM will fail!")


def test_ssim_on_different_sizes():
    """Demonstrate SSIM failure on different sized matrices"""
    from skimage.metrics import structural_similarity as ssim
    
    print(f"\n{'='*60}")
    print("SSIM COMPATIBILITY TEST")
    print(f"{'='*60}")
    
    # Same size - works
    matrix1 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    matrix2 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    
    try:
        score = ssim(matrix1, matrix2, channel_axis=2)
        print(f"✓ Same size (250x250): SSIM = {score:.3f}")
    except Exception as e:
        print(f"✗ Same size failed: {e}")
    
    # Different size - fails
    matrix3 = np.random.randint(0, 255, (250, 247, 3), dtype=np.uint8)
    matrix4 = np.random.randint(0, 255, (247, 243, 3), dtype=np.uint8)
    
    try:
        score = ssim(matrix3, matrix4, channel_axis=2)
        print(f"✓ Different size (250x247 vs 247x243): SSIM = {score:.3f}")
    except Exception as e:
        print(f"✗ Different size FAILED: {type(e).__name__}")
        print(f"   Error: Input images must have the same dimensions")


def propose_solutions():
    """Print proposed solutions"""
    print(f"\n{'='*60}")
    print("PROPOSED SOLUTIONS")
    print(f"{'='*60}")
    
    print("""
1. **PAD SHORTER SEQUENCES** (align to longest)
   - Pad with gaps or special character
   - Encode gaps as neutral color (e.g., gray)
   - Makes all matrices same size
   
2. **RESIZE ALL MATRICES TO SAME SIZE**
   - Use interpolation (e.g., bilinear)
   - ResizedSSIMMultiScaleVariant already does this!
   - But: may lose positional information
   
3. **ALIGN SEQUENCES FIRST, THEN BUILD MATRICES**
   - Use MSA (Multiple Sequence Alignment)
   - Keep gap characters ('-') in sequences
   - Encode gaps distinctly in RGB matrix
   
4. **USE ONLY ALIGNMENT-BASED METHODS WHEN INDELS PRESENT**
   - NW/SW naturally handle indels via gap penalties
   - Skip SSIM variants if length variation > threshold
   
5. **CROP TO MINIMUM OVERLAP REGION**
   - Find conserved regions
   - Only compare those regions
   - Loss of information but dimension consistency
""")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_indel_issue.py <fasta_file>")
        print("\nExample:")
        print("  python diagnose_indel_issue.py test_output_dna/evolved_sequences.fasta")
        sys.exit(1)
    
    fasta_path = Path(sys.argv[1])
    
    if not fasta_path.exists():
        print(f"Error: File not found: {fasta_path}")
        sys.exit(1)
    
    # Run diagnostics
    sequences, lengths = analyze_sequence_lengths(fasta_path)
    
    if max(lengths) - min(lengths) > 0:
        print(f"\n⚠️  INDELS DETECTED! Length variation = {max(lengths) - min(lengths)} bp")
        test_matrix_generation(sequences)
        test_ssim_on_different_sizes()
        propose_solutions()
    else:
        print(f"\n✓ No length variation - all sequences same length")
        print(f"  SSIM should work fine!")


if __name__ == '__main__':
    main()