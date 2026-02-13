#!/usr/bin/env python
"""Optimize hyperparameters for similarity algorithms using Bayesian optimization."""

import argparse
import sys
from pathlib import Path
import optuna
from ete3 import Tree
from biomodelml.variants.control import ControlVariant
from biomodelml.variants.resized_ssim_multiscale import ResizedSSIMMultiScaleVariant
from biomodelml.experiment import Experiment

SEED = 42


class HyperparameterOptimizer:
    """Optimize algorithm hyperparameters using Optuna."""
    
    def __init__(self, data_path: Path, seq_name: str, n_trials: int = 200):
        self.data_path = data_path
        self.seq_name = seq_name
        self.n_trials = n_trials
        self.control_tree = None
    
    def build_control_tree(self):
        """Build control tree using Clustal Omega."""
        print("Building control tree with Clustal Omega...")
        fasta_file = self.data_path / f"{self.seq_name}.fasta.N.sanitized"
        
        experiment = Experiment(
            self.data_path,
            ControlVariant(fasta_file, "N")
        ).run()
        
        tree_struct = experiment._trees[0]
        newick_ctl = tree_struct.tree.to_newick(
            labels=tree_struct.distances.names,
            include_distance=False
        )
        
        self.control_tree = Tree(newick_ctl, format=1)
        print("Control tree built successfully")
    
    def objective(self, trial):
        """Objective function for Optuna optimization."""
        params = dict(
            filter_sigma=trial.suggest_float("filter_sigma", 0.1, 1.5, step=0.1),
            filter_size=trial.suggest_int("filter_size", 3, 15)
        )
        
        # Skip if already computed
        for previous_trial in trial.study.trials:
            if (previous_trial.state == optuna.trial.TrialState.COMPLETE and 
                trial.params == previous_trial.params):
                return previous_trial.value
        
        fasta_file = self.data_path / f"{self.seq_name}.fasta.N.sanitized"
        image_path = self.data_path / "images" / self.seq_name / "full"
        
        experiment = Experiment(
            self.data_path,
            ResizedSSIMMultiScaleVariant(
                fasta_file,
                "N",
                image_path,
                **params
            )
        ).run()
        
        tree_struct = experiment._trees[0]
        newick_alg = tree_struct.tree.to_newick(
            labels=tree_struct.distances.names,
            include_distance=False
        )
        
        tree = Tree(newick_alg, format=1)
        rf_distance = self.control_tree.compare(tree, unrooted=True)["norm_rf"]
        
        print(f"Trial {trial.number}: RF distance = {rf_distance:.4f}, params = {params}")
        
        return rf_distance
    
    def optimize(self, storage_path: str = None):
        """Run optimization study."""
        if self.control_tree is None:
            self.build_control_tree()
        
        # Setup storage
        if storage_path is None:
            storage_path = f"sqlite:///{self.data_path}/{self.seq_name}.db"
        
        print(f"Creating Optuna study (storage: {storage_path})...")
        
        study = optuna.create_study(
            storage=storage_path,
            study_name="biomodelml_optimization",
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        print(f"Starting optimization with {self.n_trials} trials...")
        
        try:
            study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        finally:
            print("\n" + "="*60)
            print("OPTIMIZATION RESULTS")
            print("="*60)
            print(f"Best parameters: {study.best_params}")
            print(f"Best RF distance: {study.best_value:.4f}")
            print(f"Number of trials: {len(study.trials)}")
            print("="*60)


def main():
    """Optimize hyperparameters for phylogenetic tree algorithms."""
    parser = argparse.ArgumentParser(
        description="Optimize algorithm hyperparameters using Bayesian optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses Optuna to optimize hyperparameters of the Resized MS-SSIM algorithm
by comparing the resulting phylogenetic trees against a control tree built with
Clustal Omega. The optimization minimizes the Robinson-Foulds distance.

Examples:
  %(prog)s data/ orthologs_hemoglobin_beta
  %(prog)s data/ orthologs_myoglobin --trials 500
  %(prog)s data/ orthologs_neuroglobin --storage sqlite:///custom.db
        """
    )
    
    parser.add_argument(
        "data_path",
        help="Path to data directory containing FASTA files and images"
    )
    
    parser.add_argument(
        "seq_name",
        help="Sequence name (without extension, e.g., 'orthologs_hemoglobin_beta')"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Number of optimization trials (default: 200)"
    )
    
    parser.add_argument(
        "--storage",
        help="Optuna storage URL (default: sqlite:///data_path/seq_name.db)"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    # Validate data path
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    fasta_file = data_path / f"{args.seq_name}.fasta.N.sanitized"
    if not fasta_file.exists():
        print(f"Error: FASTA file {fasta_file} not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        optimizer = HyperparameterOptimizer(data_path, args.seq_name, args.trials)
        optimizer.optimize(args.storage)
    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
