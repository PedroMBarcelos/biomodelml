#!/usr/bin/env python
"""Summarize sequence generation metrics from an AliSim output directory.

Usage:
  python -m biomodelml.cli.summarize_sequences output_dir/

Saves `summary_stats.json` and `summary_stats.csv` under `output_dir/metadata/`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_summary(output_dir: Path) -> None:
    metadata_dir = output_dir / "metadata"
    manifest_file = metadata_dir / "generation_log.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file) as f:
        manifest = json.load(f)

    rows = []
    for job in manifest.get("jobs", []):
        fasta = output_dir.parent.joinpath(job["fasta_path"]).resolve() if job["fasta_path"].startswith("/") else output_dir.joinpath(Path(job["fasta_path"]).relative_to(output_dir.name)) if job["fasta_path"].startswith(output_dir.name) else output_dir / Path(job["fasta_path"]).relative_to(".") if job["fasta_path"].startswith(".") else Path(job["fasta_path"])
        # distances path
        distances = Path(job.get("distances_path", ""))
        if not distances.exists():
            # Try relative to repo root
            distances = output_dir / Path(job.get("distances_path", "")).relative_to(output_dir.name) if job.get("distances_path") else distances
        # Fallback: assume path inside output_dir
        if not distances.exists():
            distances = output_dir / "trees" / f"replicate_{job['job_id'].split('_')[1]}" / ("alignment_001.distances.csv")

        # Read distances CSV
        try:
            df = pd.read_csv(distances, index_col=0)
            # flatten upper triangle excluding diagonal
            vals = df.values
            # mask diagonal
            n = vals.shape[0]
            iu = np.triu_indices(n, k=1)
            pair_vals = vals[iu]
            mean_dist = float(np.nanmean(pair_vals))
            median_dist = float(np.nanmedian(pair_vals))
            min_dist = float(np.nanmin(pair_vals))
            max_dist = float(np.nanmax(pair_vals))
            std_dist = float(np.nanstd(pair_vals))
        except Exception as e:
            mean_dist = median_dist = min_dist = max_dist = std_dist = None

        rows.append({
            "job_id": job.get("job_id"),
            "fasta_path": job.get("fasta_path"),
            "distances_path": str(distances),
            "num_sequences": job.get("num_sequences"),
            "alignment_length": job.get("alignment_length"),
            "random_seed": job.get("random_seed"),
            "mean_pairwise_distance": mean_dist,
            "median_pairwise_distance": median_dist,
            "min_pairwise_distance": min_dist,
            "max_pairwise_distance": max_dist,
            "std_pairwise_distance": std_dist,
        })

    df_rows = pd.DataFrame(rows)

    # Overall summary
    overall = {}
    if not df_rows.empty:
        overall["alignment_length"] = {
            "count": int(df_rows["alignment_length"].count()),
            "min": int(df_rows["alignment_length"].min()),
            "max": int(df_rows["alignment_length"].max()),
            "mean": float(df_rows["alignment_length"].mean()),
            "median": float(df_rows["alignment_length"].median()),
            "std": float(df_rows["alignment_length"].std()),
        }
        overall["mean_pairwise_distance"] = {
            "count": int(df_rows["mean_pairwise_distance"].count()),
            "min": float(df_rows["mean_pairwise_distance"].min()),
            "max": float(df_rows["mean_pairwise_distance"].max()),
            "mean": float(df_rows["mean_pairwise_distance"].mean()),
            "median": float(df_rows["mean_pairwise_distance"].median()),
            "std": float(df_rows["mean_pairwise_distance"].std()),
        }

    # Save outputs
    out_meta = metadata_dir / "summary_stats.json"
    out_csv = metadata_dir / "summary_stats.csv"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump({"per_replicate": rows, "overall": overall}, f, indent=2)

    df_rows.to_csv(out_csv, index=False)
    print(f"Saved per-replicate CSV to {out_csv}")
    print(f"Saved overall JSON to {out_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize sequence generation outputs")
    parser.add_argument("output_dir", help="Path to the output directory (workspace-relative or absolute)")
    args = parser.parse_args()
    compute_summary(Path(args.output_dir))
