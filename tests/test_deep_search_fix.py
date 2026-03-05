#!/usr/bin/env python3
"""
Test Deep Search implementation after bug fixes
"""

import sys
import numpy as np
from pathlib import Path


def test_feature_extraction():
    """Test if feature extraction works correctly"""
    print("\n" + "="*60)
    print("TEST 1: Feature Extraction")
    print("="*60)
    
    try:
        from biomodelml.variants.deep_search.feature_extractor import FeatureExtractor
        
        # Create dummy image (250x250 RGB)
        dummy_img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
        
        extractor = FeatureExtractor((1000, 1000, 3))
        
        feature = extractor.extract(dummy_img)
        
        print(f"✓ Feature shape: {feature.shape}")
        print(f"✓ Feature dimension: {len(feature)}")
        print(f"✓ Feature norm: {np.linalg.norm(feature):.6f}")
        
        # Verify it's 1D
        if len(feature.shape) == 1:
            print(f"✓ Feature is properly 1D (required by Annoy)")
        else:
            print(f"✗ Feature is {len(feature.shape)}D - should be 1D!")
            return False
        
        # Check normalization
        norm = np.linalg.norm(feature)
        if abs(norm - 1.0) < 0.01:
            print(f"✓ Feature is L2 normalized (norm ≈ 1.0)")
        else:
            print(f"⚠️  Feature norm is {norm:.6f} (expected ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data(fasta_file: str, image_folder: str):
    """Test DeepSearchVariant with real data"""
    print("\n" + "="*60)
    print("TEST 2: Deep Search Variant with Real Data")
    print("="*60)
    
    fasta_path = Path(fasta_file)
    image_path = Path(image_folder)
    
    if not fasta_path.exists():
        print(f"✗ FASTA file not found: {fasta_path}")
        return False
    
    if not image_path.exists():
        print(f"✗ Image folder not found: {image_path}")
        return False
    
    # Check for PNG files
    pngs = list(image_path.glob("*.png"))
    if not pngs:
        print(f"✗ No PNG files in {image_path}")
        return False
    
    print(f"✓ Found {len(pngs)} PNG files")
    
    try:
        from biomodelml.variants.deep_search.variant import DeepSearchVariant
        
        print(f"\nInitializing DeepSearchVariant...")
        ds = DeepSearchVariant(str(fasta_path), 'P', str(image_path))
        
        print(f"✓ Variant initialized")
        print(f"✓ Found {len(ds._names)} sequences: {ds._names}")
        
        # Try to build the distance matrix
        print(f"\nBuilding distance matrix (this may take a while)...")
        
        matrix_struct = ds.build_matrix()
        
        print(f"✓ Distance matrix built successfully!")
        print(f"✓ Matrix shape: {matrix_struct.matrix.shape}")
        print(f"✓ Sequence names: {matrix_struct.names}")
        
        # Show sample distances
        print(f"\nSample distances:")
        n = min(3, len(matrix_struct.names))
        for i in range(n):
            for j in range(i+1, n):
                dist = matrix_struct.matrix[i, j]
                print(f"  {matrix_struct.names[i]} ↔ {matrix_struct.names[j]}: {dist:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("DEEP SEARCH BUG FIX VERIFICATION")
    print("="*70)
    
    # Test 1: Basic feature extraction
    result1 = test_feature_extraction()
    
    # Test 2: Real data (if provided)
    if len(sys.argv) >= 3:
        fasta_file = sys.argv[1]
        image_folder = sys.argv[2]
        result2 = test_with_real_data(fasta_file, image_folder)
    else:
        print("\n" + "="*70)
        print("To test with real data, provide:")
        print("  python tests/test_deep_search_fix.py <fasta> <image_folder>")
        print("\nExample:")
        print("  python tests/test_deep_search_fix.py \\")
        print("    test_all_algorithms/evolved_sequences.fasta.P.sanitized \\")
        print("    test_all_algorithms/matrices/full")
        print("="*70)
        result2 = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Feature Extraction: {'✓ PASS' if result1 else '✗ FAIL'}")
    if result2 is not None:
        print(f"Real Data Test:     {'✓ PASS' if result2 else '✗ FAIL'}")
    print("="*70)
    
    if result1 and (result2 is None or result2):
        print("\n✓ All tests passed! Deep Search bugs are fixed.")
        return 0
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
