#!/usr/bin/env python3
"""
Direct comparison: Baseline vs Phase 1 Improvements

Tests optical flow with and without Phase 1 improvements on the same dataset.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Import biomodelml
from biomodelml.variants.optical_flow import OpticalFlowVariant

# Use existing evolved sequences
fasta_file = "tests/optflow_validation_v2/evolved_sequences.fasta.P.sanitized"
image_folder = "tests/optflow_validation_v2/matrices"

print("=" * 70)
print("OPTICAL FLOW: BASELINE vs PHASE 1 IMPROVEMENTS")
print("=" * 70)
print(f"\nDataset: {fasta_file}")
print(f"Image folder: {image_folder}")

# Configuration 1: Baseline (current demo default)
print("\n" + "-" * 70)
print("Configuration 1: BASELINE (No Improvements)")
print("-" * 70)
print("  profile: None (levels=3, default Farneback)")
print("  magnitude_threshold: 0.0 (no filtering)")
print("  diagonal_ribbon_width: None (full matrix)")

baseline = OpticalFlowVariant(
    fasta_file, "P", image_folder,
    profile=None,
    magnitude_threshold=0.0,
    diagonal_ribbon_width=None
)

start = time.time()
result_baseline = baseline.build_matrix()
time_baseline = time.time() - start

print(f"\n  ✓ Completed in {time_baseline:.2f}s")
print(f"  Matrix shape: {result_baseline.matrix.shape}")
print(f"  Distance range: [{result_baseline.matrix.min():.6e}, {result_baseline.matrix.max():.6e}]")
print(f"  Mean distance: {result_baseline.matrix.mean():.6e}")
print(f"  Std deviation: {result_baseline.matrix.std():.6e}")

# Get off-diagonal distances
n = result_baseline.matrix.shape[0]
off_diag_baseline = []
for i in range(n):
    for j in range(i+1, n):
        off_diag_baseline.append(result_baseline.matrix[i, j])

print(f"  Off-diagonal stats:")
print(f"    Min: {np.min(off_diag_baseline):.6e}")
print(f"    Max: {np.max(off_diag_baseline):.6e}")
print(f"    Mean: {np.mean(off_diag_baseline):.6e}")
print(f"    Median: {np.median(off_diag_baseline):.6e}")

# Configuration 2: Phase 1 Improvements (Recommended)
print("\n" + "-" * 70)
print("Configuration 2: PHASE 1 IMPROVEMENTS (Recommended)")
print("-" * 70)
print("  profile: 'accurate' (levels=5)")
print("  magnitude_threshold: 0.0 (no filtering for small signals)")
print("  diagonal_ribbon_width: 50 (focus on diagonal)")

improved = OpticalFlowVariant(
    fasta_file, "P", image_folder,
    profile='accurate',
    magnitude_threshold=0.0,  # Keep at 0 since signals are already tiny
    diagonal_ribbon_width=50
)

start = time.time()
result_improved = improved.build_matrix()
time_improved = time.time() - start

print(f"\n  ✓ Completed in {time_improved:.2f}s")
print(f"  Matrix shape: {result_improved.matrix.shape}")
print(f"  Distance range: [{result_improved.matrix.min():.6e}, {result_improved.matrix.max():.6e}]")
print(f"  Mean distance: {result_improved.matrix.mean():.6e}")
print(f"  Std deviation: {result_improved.matrix.std():.6e}")

# Get off-diagonal distances
off_diag_improved = []
for i in range(n):
    for j in range(i+1, n):
        off_diag_improved.append(result_improved.matrix[i, j])

print(f"  Off-diagonal stats:")
print(f"    Min: {np.min(off_diag_improved):.6e}")
print(f"    Max: {np.max(off_diag_improved):.6e}")
print(f"    Mean: {np.mean(off_diag_improved):.6e}")
print(f"    Median: {np.median(off_diag_improved):.6e}")

# Configuration 3: Sensitive profile (Maximum tracking)
print("\n" + "-" * 70)
print("Configuration 3: SENSITIVE PROFILE (Maximum Sensitivity)")
print("-" * 70)
print("  profile: 'sensitive' (levels=7, winsize=21)")
print("  magnitude_threshold: 0.0")
print("  diagonal_ribbon_width: None (full for testing)")

sensitive = OpticalFlowVariant(
    fasta_file, "P", image_folder,
    profile='sensitive',
    magnitude_threshold=0.0,
    diagonal_ribbon_width=None  # Use full matrix to compare with baseline
)

start = time.time()
result_sensitive = sensitive.build_matrix()
time_sensitive = time.time() - start

print(f"\n  ✓ Completed in {time_sensitive:.2f}s")
print(f"  Matrix shape: {result_sensitive.matrix.shape}")
print(f"  Distance range: [{result_sensitive.matrix.min():.6e}, {result_sensitive.matrix.max():.6e}]")
print(f"  Mean distance: {result_sensitive.matrix.mean():.6e}")
print(f"  Std deviation: {result_sensitive.matrix.std():.6e}")

# Get off-diagonal distances
off_diag_sensitive = []
for i in range(n):
    for j in range(i+1, n):
        off_diag_sensitive.append(result_sensitive.matrix[i, j])

print(f"  Off-diagonal stats:")
print(f"    Min: {np.min(off_diag_sensitive):.6e}")
print(f"    Max: {np.max(off_diag_sensitive):.6e}")
print(f"    Mean: {np.mean(off_diag_sensitive):.6e}")
print(f"    Median: {np.median(off_diag_sensitive):.6e}")

# Summary comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print("\n📊 Computation Time:")
print(f"  Baseline:   {time_baseline:.2f}s")
print(f"  Improved:   {time_improved:.2f}s  ({(time_improved/time_baseline - 1)*100:+.1f}%)")
print(f"  Sensitive:  {time_sensitive:.2f}s  ({(time_sensitive/time_baseline - 1)*100:+.1f}%)")

print("\n📏 Off-Diagonal Distance Statistics:")
print(f"  Baseline:   mean={np.mean(off_diag_baseline):.6e}, std={np.std(off_diag_baseline):.6e}")
print(f"  Improved:   mean={np.mean(off_diag_improved):.6e}, std={np.std(off_diag_improved):.6e}")
print(f"  Sensitive:  mean={np.mean(off_diag_sensitive):.6e}, std={np.std(off_diag_sensitive):.6e}")

print("\n🎯 Distance Dynamic Range (Max/Min ratio):")
baseline_range = np.max(off_diag_baseline) / max(np.min(off_diag_baseline), 1e-20)
improved_range = np.max(off_diag_improved) / max(np.min(off_diag_improved), 1e-20)
sensitive_range = np.max(off_diag_sensitive) / max(np.min(off_diag_sensitive), 1e-20)
print(f"  Baseline:   {baseline_range:.2f}x")
print(f"  Improved:   {improved_range:.2f}x")
print(f"  Sensitive:  {sensitive_range:.2f}x")

print("\n💡 Key Findings:")
print("  • All configurations detect relative differences:")
if np.std(off_diag_baseline) > 0:
    cv_baseline = np.std(off_diag_baseline) / np.mean(off_diag_baseline)
    cv_improved = np.std(off_diag_improved) / np.mean(off_diag_improved)
    cv_sensitive = np.std(off_diag_sensitive) / np.mean(off_diag_sensitive)
    print(f"    - Baseline CV: {cv_baseline:.3f}")
    print(f"    - Improved CV: {cv_improved:.3f}")
    print(f"    - Sensitive CV: {cv_sensitive:.3f}")
    
    # Diagonal masking should reduce computation time
    speedup = (1 - time_improved/time_baseline) * 100
    if speedup > 0:
        print(f"  • Diagonal masking speedup: {speedup:.1f}%")
    
    # Check if dynamic range improved
    if improved_range > baseline_range:
        improvement = ((improved_range / baseline_range) - 1) * 100
        print(f"  • Dynamic range improved: +{improvement:.1f}%")

print("\n⚠️  Note: Absolute distances are very small (1e-05 to 1e-09)")
print("   This is expected for highly similar sequences (~100 AA with <5 mutations)")
print("   Tree building relies on RELATIVE distances, not absolute magnitude")

print("\n✅ Phase 1 validation complete!")
print("=" * 70)
