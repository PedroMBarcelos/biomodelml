from __future__ import annotations
import traceback
import pandas
import dataclasses
import numpy
from pathlib import Path
from typing import Iterable
from biomodelml.variants.variant import Variant
from biomodelml.structs import TreeStruct, ImgDebug
from biotite.sequence import phylo
from matplotlib import pyplot
from Bio import Phylo
from io import StringIO
from matplotlib.figure import Figure

try:
    from IPython.display import display, Image as IPImage
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

class Experiment:
    def __init__(self, output_path: Path, *variants: Iterable[Variant]):
        self._output_path = output_path
        self._variants = variants
        self._trees = []

    def run(self) -> Experiment:
        self._trees = []
        for variant in self._variants:
            try:
                distances = variant.build_matrix()
                tree = phylo.neighbor_joining(distances.matrix)
                self._trees.append(
                    TreeStruct(name=variant.name, distances=distances, tree=tree))
            except Exception:
                print(traceback.format_exc())
        return self
    
    def run_and_save(self):
        successful_trees = []
        failed_variants = []
        for variant in self._variants:
            try:
                distances = variant.build_matrix()
                tree_struct = TreeStruct(
                    name=variant.name, distances=distances,
                    tree=phylo.neighbor_joining(distances.matrix))
                self._save_all(None, tree_struct)
                successful_trees.append(tree_struct)
                print(f"{variant.name} done!")
            except Exception:
                failed_variants.append(getattr(variant, "name", "unknown"))
                print(traceback.format_exc())        
        self._save_benchmark_summary(successful_trees, failed_variants)

    @staticmethod
    def _vectorize_upper_triangle(matrix: numpy.ndarray) -> numpy.ndarray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return numpy.array([], dtype=numpy.float64)
        tri = numpy.triu_indices(matrix.shape[0], k=1)
        return matrix[tri].astype(numpy.float64, copy=False)

    @staticmethod
    def _matrix_against_reference(matrix: numpy.ndarray, reference: numpy.ndarray):
        vec = Experiment._vectorize_upper_triangle(matrix)
        ref = Experiment._vectorize_upper_triangle(reference)
        if vec.size == 0 or ref.size == 0 or vec.size != ref.size:
            return {
                "mae": numpy.nan,
                "rmse": numpy.nan,
                "pearson_r": numpy.nan,
            }

        delta = vec - ref
        mae = float(numpy.mean(numpy.abs(delta)))
        rmse = float(numpy.sqrt(numpy.mean(delta * delta)))

        vec_std = float(numpy.std(vec))
        ref_std = float(numpy.std(ref))
        if vec_std > 0.0 and ref_std > 0.0:
            pearson_r = float(numpy.corrcoef(vec, ref)[0, 1])
        else:
            pearson_r = numpy.nan

        return {
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson_r,
        }

    def _save_benchmark_summary(self, successful_trees: list[TreeStruct], failed_variants: list[str]):
        rows = []
        siamese_ref = None
        for tree_struct in successful_trees:
            if "siamese" in tree_struct.name.lower():
                siamese_ref = tree_struct.distances.matrix.astype(numpy.float64, copy=False)
                break

        for tree_struct in successful_trees:
            matrix = tree_struct.distances.matrix.astype(numpy.float64, copy=False)
            upper = self._vectorize_upper_triangle(matrix)
            if upper.size > 0:
                pair_mean = float(numpy.mean(upper))
                pair_std = float(numpy.std(upper))
                pair_min = float(numpy.min(upper))
                pair_max = float(numpy.max(upper))
            else:
                pair_mean = numpy.nan
                pair_std = numpy.nan
                pair_min = numpy.nan
                pair_max = numpy.nan

            if siamese_ref is not None and tree_struct.name.lower().find("siamese") == -1:
                vs_siamese = self._matrix_against_reference(matrix, siamese_ref)
            else:
                vs_siamese = {
                    "mae": numpy.nan,
                    "rmse": numpy.nan,
                    "pearson_r": numpy.nan,
                }

            rows.append({
                "variant": tree_struct.name,
                "n_taxa": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
                "pair_mean": pair_mean,
                "pair_std": pair_std,
                "pair_min": pair_min,
                "pair_max": pair_max,
                "diag_mean": float(numpy.mean(numpy.diag(matrix))) if matrix.ndim == 2 else numpy.nan,
                "symmetry_max_abs": float(numpy.max(numpy.abs(matrix - matrix.T))) if matrix.ndim == 2 else numpy.nan,
                "vs_siamese_mae": vs_siamese["mae"],
                "vs_siamese_rmse": vs_siamese["rmse"],
                "vs_siamese_pearson_r": vs_siamese["pearson_r"],
                "status": "ok",
            })

        for variant_name in failed_variants:
            rows.append({
                "variant": variant_name,
                "n_taxa": 0,
                "pair_mean": numpy.nan,
                "pair_std": numpy.nan,
                "pair_min": numpy.nan,
                "pair_max": numpy.nan,
                "diag_mean": numpy.nan,
                "symmetry_max_abs": numpy.nan,
                "vs_siamese_mae": numpy.nan,
                "vs_siamese_rmse": numpy.nan,
                "vs_siamese_pearson_r": numpy.nan,
                "status": "failed",
            })

        if rows:
            pandas.DataFrame(rows).to_csv(self._output_path / "benchmark_summary.csv", index=False)

    def _save_distance_matrix(self, tree_struct: TreeStruct):
        pandas.DataFrame(
            data=tree_struct.distances.matrix,
            index=tree_struct.distances.names,
            columns=tree_struct.distances.names).to_csv(
                self._output_path / f"{tree_struct.name}.csv"
            )

    def _save_img_debugs(self, tree_struct: TreeStruct):
        if tree_struct.distances.img_debugs:
            debuglist = ["img1", "img2"] + [field.name for field in dataclasses.fields(ImgDebug)]
            debug = f"{','.join(debuglist)}\n"
            for items in tree_struct.distances.img_debugs:
                img1, img2, debugs = dataclasses.astuple(items)
                for d in debugs:
                    debug += f"{img1},{img2},{','.join(d)}\n"
            with open(self._output_path / f"{tree_struct.name}.map", "w") as f:
                f.write(debug)

    def _save_align(self, tree_struct: TreeStruct):
        if tree_struct.distances.align:
            align = ""
            for i, seq in enumerate(tree_struct.distances.align.get_gapped_sequences()):
                align += f">{tree_struct.distances.names[i]}\n"
                align += f"{seq}\n"
            with open(self._output_path / f"{tree_struct.name}.fasta", "w") as f:
                f.write(align)

    def _save_newick_tree(self, tree_struct: TreeStruct):
        newick = tree_struct.tree.to_newick(
            labels=tree_struct.distances.names, include_distance=False)
        with open(self._output_path / f"{tree_struct.name}.nw", "w") as f:
            f.write(newick)

    def _save_plot_tree(self, fig: Figure, tree_struct: TreeStruct):
        # Create a fresh figure for this tree
        fig = pyplot.figure(figsize=(12.0, 8.0))
        ax = fig.add_subplot(111)
        
        newick = tree_struct.tree.to_newick(include_distance=False)
        t = Phylo.read(StringIO(newick), "newick")
        t.ladderize()
        
        # Use Phylo.draw with proper configuration
        Phylo.draw(t, axes=ax, do_show=False)
        fig.suptitle(tree_struct.name, fontsize=16)
        
        # Save to file
        png_path = self._output_path / f"{tree_struct.name}.png"
        fig.savefig(png_path, bbox_inches='tight', dpi=100)
        
        # Close the figure
        pyplot.close(fig)
    
    def _draw_tree_with_labels(self, ax, tree, labels):
        """Draw a phylogenetic tree on the given axis with proper branch rendering."""
        def get_label(clade):
            """Get the label for a clade."""
            if clade.name and clade.name.isdigit():
                try:
                    return labels[int(clade.name)]
                except (ValueError, IndexError):
                    return clade.name or ""
            return clade.name or ""
        
        # Calculate tree depth and create scaling
        terminals = tree.get_terminals()
        max_depth = max(tree.distance(leaf) for leaf in terminals) if terminals else 1.0
        max_depth = max(max_depth, 1.0)  # Ensure minimum depth
        
        # Set up coordinates for each clade  
        def calc_coords(clade, pos_x=0.0, pos_y=0.0, depth_scale=1.0):
            """Calculate coordinates for all clades recursively."""
            if clade.is_terminal():
                clade.y = pos_y
                clade.x = pos_x
            else:
                # Calculate position for child clades
                num_children = len(clade.clades)
                if num_children > 0:
                    y_spread = 1.0 / (num_children + 1)
                    for i, child in enumerate(clade.clades):
                        child_y = pos_y - 0.5 + (i + 1) * y_spread
                        # Scale branch length for visualization
                        branch_len = (child.branch_length or 0.1) * depth_scale
                        child_x = pos_x - branch_len
                        calc_coords(child, child_x, child_y, depth_scale)
                clade.x = pos_x
                if num_children > 0:
                    clade.y = sum(child.y for child in clade.clades) / num_children
                else:
                    clade.y = pos_y
        
        # Scale depth to fit nicely in the plot
        depth_scale = 2.0 / max_depth if max_depth > 0 else 1.0
        calc_coords(tree.clade, pos_x=0.0, pos_y=0.5, depth_scale=depth_scale)
        
        # Draw branches with thicker lines
        def draw_branches(clade):
            """Draw lines for the clade and its children."""
            for child in clade.clades:
                ax.plot([clade.x, child.x], [clade.y, child.y], 'k-', linewidth=1.5, solid_capstyle='round')
                draw_branches(child)
        
        draw_branches(tree.clade)
        
        # Draw labels at terminal nodes
        for leaf in terminals:
            label = get_label(leaf)
            if label:
                ax.text(leaf.x, leaf.y, f"  {label}", fontsize=9, ha='left', va='center')
        
        # Set axis limits with padding
        xs = [leaf.x for leaf in terminals]
        ys = [leaf.y for leaf in terminals]
        x_min, x_max = min(xs) if xs else 0, max(xs) if xs else 1
        y_min, y_max = min(ys) if ys else 0, max(ys) if ys else 1
        
        x_padding = (x_max - x_min) * 0.15 if x_max > x_min else 0.3
        y_padding = 0.1
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    def _save_all(self, fig: Figure, tree_struct: TreeStruct):
        self._save_img_debugs(tree_struct)
        self._save_align(tree_struct)
        self._save_newick_tree(tree_struct)
        self._save_distance_matrix(tree_struct)
        self._save_plot_tree(fig, tree_struct)

    def save(self):
        fig = pyplot.figure(figsize=(12.0, 12.0))
        for tree_struct in self._trees:
            self._save_all(fig, tree_struct)