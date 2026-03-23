import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os
import sys
from tqdm import tqdm
import importlib
import numpy as np
import matplotlib.pyplot as plt


# Ensure local project package is imported when this file is run directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from biomodelml import datasets
from biomodelml.models import SiameseRegressor

# Force reload of the datasets module to ensure the latest version is used
importlib.reload(datasets)
from biomodelml.datasets import SiameseEvolutionDataset


def evaluate(args):
    """
    Evaluate a trained Siamese Regressor on the test set.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    print("Loading dataset...")
    full_dataset = SiameseEvolutionDataset(cache_dir=args.cache_dir, seq_type=args.seq_type)

    # 3. Split dataset (same split as training with same seed for reproducibility)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Same seed as training
    )

    print(f"Test set size: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")

    # 4. Create Test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5. Load Model
    print(f"Loading model from {args.model_path}...")
    model = SiameseRegressor(
        backbone=args.backbone,
        freeze_backbone=False  # Doesn't matter for evaluation
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 6. Evaluate
    print("Evaluating on test set...")
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for (img1, img2), dist in tqdm(test_loader, desc="Testing"):
            img1, img2 = img1.to(device), img2.to(device)
            dist = dist.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, dist)
            test_loss += loss.item()

            # Store predictions and targets
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(dist.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader)

    # 7. Calculate Metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # Correlation coefficient
    correlation = np.corrcoef(predictions, targets)[0, 1]

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    # 8. Print Results
    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"MAE:             {mae:.6f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"Correlation:     {correlation:.4f}")
    print(f"R² Score:        {r2_score:.4f}")
    print("=" * 60)

    # 9. Save Results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, 'test_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Test Set Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Dataset: {args.cache_dir}\n")
            f.write(f"Test samples: {len(test_dataset)}\n\n")
            f.write(f"Test Loss (MSE): {avg_test_loss:.6f}\n")
            f.write(f"MAE:             {mae:.6f}\n")
            f.write(f"RMSE:            {rmse:.6f}\n")
            f.write(f"Correlation:     {correlation:.4f}\n")
            f.write(f"R² Score:        {r2_score:.4f}\n")
        print(f"\nMetrics saved to {metrics_path}")

        # Save predictions vs targets
        predictions_path = os.path.join(args.output_dir, 'predictions.npz')
        np.savez(predictions_path, predictions=predictions, targets=targets)
        print(f"Predictions saved to {predictions_path}")

        # Create scatter plot
        if args.plot:
            plt.figure(figsize=(8, 8))
            plt.scatter(targets, predictions, alpha=0.5, s=10)
            plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                     'r--', lw=2, label='Perfect prediction')
            plt.xlabel('True Distance', fontsize=12)
            plt.ylabel('Predicted Distance', fontsize=12)
            plt.title(f'Test Set: Predictions vs True Values\nR²={r2_score:.4f}, Correlation={correlation:.4f}',
                      fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(args.output_dir, 'predictions_plot.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to {plot_path}")
            plt.close()

    return {
        'test_loss': avg_test_loss,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'r2_score': r2_score
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Siamese Regressor on the test set."
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Directory containing the cached dataset')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0'],
                        help='CNN backbone (must match training)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate scatter plot of predictions')

    args = parser.parse_args()
    evaluate(args)
