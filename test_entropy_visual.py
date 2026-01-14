#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def adjusted_entropy(proportions):
    """Compute adjusted entropy as per paper.
    E_s = [H - log(k)] / log(k)
    Range: [-1, 0] where 0 = uniform (max diversity), -1 = single type (min diversity)
    """
    k = len(proportions)
    if k <= 1:
        return -1.0
    p_nonzero = proportions[proportions > 0]
    H = -(p_nonzero * np.log(p_nonzero)).sum()
    return (H - np.log(k)) / np.log(k)

# Create test distributions (5 cell types)
n_types = 5

# Test 1: Uniform distribution (max diversity)
uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Test 2: Two cell types dominate
two_dominated = np.array([0.45, 0.45, 0.05, 0.03, 0.02])

# Test 3: One cell type dominates strongly
one_dominated = np.array([0.90, 0.05, 0.03, 0.01, 0.01])

# Test 4: Only ONE cell type present (should be exactly -1)
single_type = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

# Calculate entropies
entropy_uniform = adjusted_entropy(uniform)
entropy_two = adjusted_entropy(two_dominated)
entropy_one = adjusted_entropy(one_dominated)
entropy_single = adjusted_entropy(single_type)

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Top row: bar plots of proportions
cell_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5']
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']

axes[0, 0].bar(cell_types, uniform, color=colors, alpha=0.7)
axes[0, 0].set_title(f'Uniform Distribution\nEntropy = {entropy_uniform:.3f}', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Proportion', fontsize=11)
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(cell_types, two_dominated, color=colors, alpha=0.7)
axes[0, 1].set_title(f'Two Types Dominate\nEntropy = {entropy_two:.3f}', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(0, 1.0)
axes[0, 1].tick_params(axis='x', rotation=45)

axes[0, 2].bar(cell_types, one_dominated, color=colors, alpha=0.7)
axes[0, 2].set_title(f'One Type Dominates\nEntropy = {entropy_one:.3f}', fontsize=12, fontweight='bold')
axes[0, 2].set_ylim(0, 1.0)
axes[0, 2].tick_params(axis='x', rotation=45)

axes[0, 3].bar(cell_types, single_type, color=colors, alpha=0.7)
axes[0, 3].set_title(f'Single Type Only\nEntropy = {entropy_single:.3f}', fontsize=12, fontweight='bold')
axes[0, 3].set_ylim(0, 1.0)
axes[0, 3].tick_params(axis='x', rotation=45)

# Bottom row: entropy scale visualization
entropies = [entropy_uniform, entropy_two, entropy_one, entropy_single]
labels = ['Uniform', 'Two\nDominate', 'One\nDominates', 'Single\nType']

for i in range(4):
    ax = axes[1, i]
    
    # Draw entropy scale
    ax.barh([0], [entropies[i]], height=0.5, color='steelblue', alpha=0.8)
    
    # Reference lines
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Max diversity')
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Min diversity')
    
    ax.set_xlim(-1.05, 0.05)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Adjusted Entropy', fontsize=11)
    ax.set_yticks([])
    ax.set_title(labels[i], fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    if i == 0:
        ax.legend(loc='lower left', fontsize=8)

plt.suptitle('Adjusted Entropy Formula Demonstration\nE_s = [H - log(k)] / log(k)    (Range: -1 to 0)', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/scratch/easmit31/variability/entropy_formula_test.png', dpi=150, bbox_inches='tight')
print("Saved visualization to /scratch/easmit31/variability/entropy_formula_test.png")

# Print summary
print("\n" + "="*60)
print("ENTROPY FORMULA TEST RESULTS")
print("="*60)
print(f"Uniform distribution:       {entropy_uniform:.4f}  (Maximum diversity)")
print(f"Two types dominate:         {entropy_two:.4f}")
print(f"One type dominates:         {entropy_one:.4f}")
print(f"Single type only:           {entropy_single:.4f}  (Minimum diversity - exactly -1)")
print(f"\nScale: 0 = maximum diversity, -1 = minimum diversity")
print("="*60)
