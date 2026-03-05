#!/usr/bin/env python3
from ete3 import Tree

# Load trees
true = Tree("test_output_final/true_tree.nw", format=1)
inferred = Tree("test_output_final/Global with Needleman-Wunsch.nw", format=1)

print("TRUE TREE:")
print("  Newick:", true.write(format=1)[:100])
print(true.get_ascii(show_internal=False))

print("\nINFERRED TREE (Needleman-Wunsch):")
print("  Newick:", inferred.write(format=1)[:100])
print(inferred.get_ascii(show_internal=False))

print("\n" + "="*70)
print("STRUCTURAL COMPARISON:")
print("="*70)
print(f"True tree - Root children: {len(true.children)} (binary tree)")
print(f"Inferred tree - Root children: {len(inferred.children)}", end="")

# Check if it's a polytomy
if len(inferred.children) > 2:
    print(" ⚠️ POLYTOMY (multifurcation)!")
    print("\n🔍 EXPLANATION:")
    print("   The inferred tree has an unresolved relationship at the root.")
    print("   This means biomodelml couldn't confidently determine which")
    print("   groups should be sister clades, so it left them as a polytomy.")
    print("\n   RF distance = 0 because all RESOLVED relationships match.")
    print("   But visually they look different due to the polytomy structure.")
else:
    print()

# Show bipartitions for comparison
def get_bps(tree):
    bps = []
    all_leaves = set(tree.get_leaf_names())
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            desc = frozenset(node.get_leaf_names())
            if len(desc) <= len(all_leaves) / 2:
                bps.append(sorted(desc))
            else:
                bps.append(sorted(all_leaves - desc))
    return sorted([tuple(bp) for bp in bps])

print(f"\nTrue tree bipartitions: {len(get_bps(true))}")
for bp in get_bps(true):
    print(f"  {bp}")

print(f"\nInferred tree bipartitions: {len(get_bps(inferred))}")
for bp in get_bps(inferred):
    print(f"  {bp}")
