#!/usr/bin/env python3
import numpy as np
import pandas as pd

def adjusted_entropy_old(proportions):
    """OLD version - standard normalized entropy [0, 1]"""
    if len(proportions) <= 1:
        return 0.0
    p_nonzero = proportions[proportions > 0]
    H = -(p_nonzero * np.log(p_nonzero)).sum()
    return H / np.log(len(proportions))

def adjusted_entropy_correct(proportions):
    """CORRECTED version from paper:
    E_s = [-Σ(p_i * log(p_i)) - log(k)] / log(k)
        = [-Σ(p_i * log(p_i)) / log(k)] - 1
    Range: [-1, 0] where 0 = uniform (max diversity), -1 = single type (min diversity)
    """
    k = len(proportions)
    if k <= 1:
        return -1.0  # Single category = minimum diversity
    
    # Filter out zeros to avoid log(0)
    p_nonzero = proportions[proportions > 0]
    
    # Calculate Shannon entropy
    H = -(p_nonzero * np.log(p_nonzero)).sum()
    
    # Apply the formula: (H - log(k)) / log(k) = H/log(k) - 1
    return (H - np.log(k)) / np.log(k)

print("=" * 60)
print("Testing CORRECTED Entropy Formula from Paper")
print("E_s = [H - log(k)] / log(k), where H = -Σ(p_i * log(p_i))")
print("Range: [-1, 0] where 0 = max diversity, -1 = min diversity")
print("=" * 60)

# Test 1: Uniform distribution (maximum entropy)
print("\nTest 1: Uniform (5 categories) - MAXIMUM DIVERSITY")
uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
print(f"Proportions: {uniform}")
print(f"OLD formula [0,1]:     {adjusted_entropy_old(uniform):.4f}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(uniform):.4f}")
print(f"Expected: 0.0 (maximum diversity in paper's scale)")

# Test 2: Completely skewed - MINIMUM DIVERSITY
print("\nTest 2: Completely skewed (one type dominates) - MINIMUM DIVERSITY")
skewed = np.array([0.9999, 0.00025, 0.00025, 0.00025, 0.00025])
print(f"Proportions: {skewed}")
print(f"OLD formula [0,1]:     {adjusted_entropy_old(skewed):.4f}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(skewed):.4f}")
print(f"Expected: ~-1.0 (minimum diversity in paper's scale)")

# Test 3: Single category
print("\nTest 3: Single category (all one type) - MINIMUM DIVERSITY")
single = np.array([1.0])
print(f"Proportions: {single}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(single):.4f}")
print(f"Expected: -1.0")

# Test 4: Two categories, equal
print("\nTest 4: Two categories, equal - MAXIMUM for 2 categories")
two_equal = np.array([0.5, 0.5])
print(f"Proportions: {two_equal}")
print(f"OLD formula [0,1]:     {adjusted_entropy_old(two_equal):.4f}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(two_equal):.4f}")
print(f"Expected: 0.0 (uniform)")

# Test 5: 10 categories, uniform
print("\nTest 5: 10 categories, uniform")
ten_uniform = np.ones(10) / 10
print(f"OLD formula [0,1]:     {adjusted_entropy_old(ten_uniform):.4f}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(ten_uniform):.4f}")
print(f"Expected: 0.0 (uniform)")

# Test 6: Intermediate
print("\nTest 6: Intermediate (5 categories, uneven)")
intermediate = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
print(f"Proportions: {intermediate}")
print(f"OLD formula [0,1]:     {adjusted_entropy_old(intermediate):.4f}")
print(f"CORRECTED [-1,0]:      {adjusted_entropy_correct(intermediate):.4f}")
print(f"Expected: between -1 and 0")

print("\n" + "=" * 60)
print("Key Insight: Paper uses REVERSED scale!")
print("  Standard entropy: 0 (min diversity) → 1 (max diversity)")
print("  Paper's E_s:     -1 (min diversity) → 0 (max diversity)")
print("=" * 60)
