"""
Training Script for 4-Channel Siamese Regressor

This script trains the Siamese regressor with 4-channel input (RGB + Mask) for
evolutionary distance prediction. Features include:

- Support for on-the-fly generation or cached HDF5 datasets
- Train/validation/test split (70/15/15)
- Early stopping with patience
- Model checkpointing (saves best model)
- Comprehensive metrics: MSE, MAE, RMSE, Pearson correlation
- GPU optimization with multi-worker dataloaders

Phase 3 Implementation: Basic training loop without curriculum learning
(Curriculum learning will be added in Phase 5)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from biomodelml.simulation import get_generator
from biomodelml.datasets_4ch import SiameseEvolutionDataset4Channel
from biomodelml.models_4ch import SiameseRegressor4Channel


DEFAULT_GTR_RATES = (1.0, 2.0, 1.0, 1.0, 2.0, 1.0)
DEFAULT_BASE_FREQS = (0.25, 0.25, 0.25, 0.25)


def evaluate_on_test_set(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set (final evaluation after training).

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        dict: Dictionary with test metrics (loss, MAE, RMSE, correlation, R²)
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    model.eval()
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

    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # Pearson correlation coefficient
    pred_mean = predictions.mean()
    target_mean = targets.mean()
    numerator = ((predictions - pred_mean) * (targets - target_mean)).sum()
    denominator = (np.sqrt(((predictions - pred_mean) ** 2).sum()) *
                   np.sqrt(((targets - target_mean) ** 2).sum()))
    correlation = numerator / denominator if denominator != 0 else 0.0

    # R² score
    ss_tot = ((targets - target_mean) ** 2).sum()
    ss_res = ((targets - predictions) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f"\nTest Set Results:")
    print(f"  Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"  MAE:             {mae:.6f}")
    print(f"  RMSE:            {rmse:.6f}")
    print(f"  Pearson R:       {correlation:.4f}")
    print(f"  R²:              {r_squared:.4f}")
    print("=" * 70)

    return {
        'test_loss': avg_test_loss,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'r_squared': r_squared
    }


def train(args):
    """
    Main training loop for the 4-Channel Siamese Regressor.

    This function handles:
    - Dataset loading (cached or on-the-fly)
    - Train/val/test split
    - Training loop with early stopping
    - Model checkpointing
    - Optional test set evaluation
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"4-Channel Siamese Regressor Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Mask Init Mode: {args.mask_init_mode}")

    # 2. Initialize Dataset
    print("\nInitializing 4-channel dataset...")

    if args.cache_file:
        # Load from HDF5 cache
        full_dataset = SiameseEvolutionDataset4Channel(
            cache_file=args.cache_file,
            max_len=args.max_len
        )
        print(f"Loaded cached dataset from: {args.cache_file}")

    else:
        # On-the-fly generation
        print("No cache file specified. Generating data on-the-fly.")
        generator = get_generator(
            args.seq_type,
            seq_len=args.seq_len,
            max_len=args.max_len,
            model_name=args.model_name,
            kappa=args.kappa,
            gtr_rates=tuple(args.gtr_rates),
            base_freqs=tuple(args.base_freqs),
            gamma_alpha=args.gamma_alpha,
            gamma_categories=args.gamma_categories,
            p_invariant=args.p_invariant,
            indel_rate=args.indel_rate,
            indel_size=args.indel_size,
        )
        full_dataset = SiameseEvolutionDataset4Channel(
            generator=generator,
            num_samples=args.num_samples,
            max_len=args.max_len,
            seq_type=args.seq_type,
            curriculum_mode='none'  # No curriculum in Phase 3
        )

    # 3. Split dataset: 70% train, 15% validation, 15% test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    print(f"\nDataset split (total: {total_size}):")
    print(f"  Training:   {train_size:6d} ({train_size/total_size*100:.1f}%)")
    print(f"  Validation: {val_size:6d} ({val_size/total_size*100:.1f}%)")
    print(f"  Test:       {test_size:6d} ({test_size/total_size*100:.1f}%)")
    print(f"\nNote: Test set is reserved for final evaluation only.")

    # 4. Create DataLoaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # 5. Initialize Model, Loss, and Optimizer
    print(f"\nInitializing 4-channel model...")
    model = SiameseRegressor4Channel(
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=not args.unfreeze_backbone,
        mask_init_mode=args.mask_init_mode
    ).to(device)

    # Print mask channel stats
    stats = model.get_mask_channel_stats()
    print(f"Initial mask channel weights: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    # Loss function
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'huber':
        criterion = torch.nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    print(f"Loss function: {args.loss.upper()}")

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_correlation': []
    }

    # 6. Training Loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        for (img1, img2), dist in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            img1, img2 = img1.to(device), img2.to(device)
            dist = dist.to(device)

            optimizer.zero_grad()

            outputs = model(img1, img2)
            loss = criterion(outputs, dist)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_predictions = []
        all_val_targets = []

        with torch.no_grad():
            for (img1, img2), dist in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]"):
                img1, img2 = img1.to(device), img2.to(device)
                dist = dist.to(device)

                outputs = model(img1, img2)
                loss = criterion(outputs, dist)
                val_loss += loss.item()

                all_val_predictions.extend(outputs.cpu().numpy().flatten())
                all_val_targets.extend(dist.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Calculate validation correlation
        val_preds = np.array(all_val_predictions)
        val_targets = np.array(all_val_targets)
        pred_mean = val_preds.mean()
        target_mean = val_targets.mean()
        numerator = ((val_preds - pred_mean) * (val_targets - target_mean)).sum()
        denominator = (np.sqrt(((val_preds - pred_mean) ** 2).sum()) *
                       np.sqrt(((val_targets - target_mean) ** 2).sum()))
        val_correlation = numerator / denominator if denominator != 0 else 0.0
        history['val_correlation'].append(val_correlation)

        print(f"Epoch [{epoch+1:3d}/{args.num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Val Corr: {val_correlation:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the best model
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            model_path = os.path.join(args.save_dir, f'siamese_4ch_{args.backbone}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  → Best model saved to {model_path}")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={args.patience})")
                break

    print("\n" + "=" * 70)
    print(f"Training Completed ({epoch+1} epochs)")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Check mask channel evolution
    final_stats = model.get_mask_channel_stats()
    print(f"\nMask channel weight evolution:")
    print(f"  Initial: mean={stats['mean']:.6f}, std={stats['std']:.6f}, abs_mean={stats['abs_mean']:.6f}")
    print(f"  Final:   mean={final_stats['mean']:.6f}, std={final_stats['std']:.6f}, abs_mean={final_stats['abs_mean']:.6f}")

    if abs(final_stats['abs_mean']) < 1e-5:
        print(f"  ⚠ Warning: Mask channel weights are still near zero. Model may not be using the mask.")
    else:
        print(f"  ✓ Mask channel is being used by the model.")

    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Correlation\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},"
                    f"{history['val_loss'][i]:.6f},{history['val_correlation'][i]:.6f}\n")
    print(f"\nTraining history saved to {history_path}")

    # 7. Evaluate on test set if requested
    if args.evaluate_test:
        print("\nLoading best model for test evaluation...")
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate_on_test_set(model, test_loader, criterion, device)

        # Save test metrics
        test_metrics_path = os.path.join(args.save_dir, 'test_metrics.txt')
        with open(test_metrics_path, 'w') as f:
            f.write("Final Test Set Evaluation\n")
            f.write("=" * 40 + "\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
        print(f"Test metrics saved to {test_metrics_path}")

    else:
        print("\nSkipping test set evaluation (use --evaluate-test to run it).")
        print("Test set is preserved for final evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train 4-Channel Siamese Regressor for evolutionary distance prediction."
    )

    # Dataset parameters
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Path to HDF5 cached dataset file (4-channel format)')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='Number of pairs to generate (on-the-fly mode)')
    parser.add_argument('--seq-len', type=int, default=500,
                        help='Length of sequences to simulate')
    parser.add_argument('--max-len', type=int, default=550,
                        help='Maximum length for padding matrices')
    parser.add_argument('--seq-type', type=str, default='N', choices=['N', 'P'],
                        help='Sequence type: N=nucleotide, P=protein')

    # On-the-fly nucleotide evolution parameters (ignored when --cache-file is used)
    parser.add_argument('--model-name', type=str, default='gtr', choices=['gtr', 'hky85'],
                        help='Nucleotide substitution model used in on-the-fly generation')
    parser.add_argument('--kappa', type=float, default=4.0,
                        help='Transition/transversion ratio for HKY85 mode')
    parser.add_argument('--gtr-rates', type=float, nargs=6,
                        default=list(DEFAULT_GTR_RATES),
                        metavar=('AC', 'AG', 'AT', 'CG', 'CT', 'GT'),
                        help='Six exchangeability rates for GTR model')
    parser.add_argument('--base-freqs', type=float, nargs=4,
                        default=list(DEFAULT_BASE_FREQS),
                        metavar=('A', 'C', 'G', 'T'),
                        help='Base frequencies used by HKY85/GTR models')
    parser.add_argument('--gamma-alpha', type=float, default=0.5,
                        help='Alpha parameter for discrete gamma rate heterogeneity')
    parser.add_argument('--gamma-categories', type=int, default=4,
                        help='Number of discrete gamma rate categories')
    parser.add_argument('--p-invariant', type=float, default=0.1,
                        help='Proportion of invariant sites')
    parser.add_argument('--indel-rate', type=float, default=0.0005,
                        help='Indel rate for on-the-fly nucleotide evolution')
    parser.add_argument('--indel-size', type=int, default=1,
                        help='Default indel event size in on-the-fly evolution')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0'],
                        help='CNN backbone architecture')
    parser.add_argument('--mask-init-mode', type=str, default='zero',
                        choices=['zero', 'mean', 'random'],
                        help='Mask channel initialization: zero (recommended), mean, or random')
    parser.add_argument('--unfreeze-backbone', action='store_true',
                        help='If set, train the backbone (not just regression head)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber'],
                        help='Loss function: mse or huber')

    # Infrastructure parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers (0 = single-threaded)')
    parser.add_argument('--save-dir', type=str, default='models/siamese_4ch',
                        help='Directory to save trained model')

    # Evaluation parameters
    parser.add_argument('--evaluate-test', action='store_true',
                        help='If set, evaluate on test set after training')

    args = parser.parse_args()

    # Validate arguments
    if args.cache_file and not os.path.exists(args.cache_file):
        print(f"Error: Cache file not found: {args.cache_file}")
        sys.exit(1)

    if abs(sum(args.base_freqs) - 1.0) > 1e-6:
        print("Error: --base-freqs must sum to 1.0")
        sys.exit(1)

    if not (0.0 <= args.p_invariant < 1.0):
        print("Error: --p-invariant must be in [0.0, 1.0)")
        sys.exit(1)

    if args.gamma_categories < 1:
        print("Error: --gamma-categories must be >= 1")
        sys.exit(1)

    if args.indel_size < 1:
        print("Error: --indel-size must be >= 1")
        sys.exit(1)

    train(args)
