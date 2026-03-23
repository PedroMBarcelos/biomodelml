import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import sys
from tqdm import tqdm
import importlib


# Ensure local project package is imported when this file is run directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from biomodelml.simulation import SyntheticEvolutionGenerator
from biomodelml import datasets
from biomodelml.models import SiameseRegressor

# Force reload of the datasets module to ensure the latest version is used
importlib.reload(datasets)
print(f"Loading dataset module from: {datasets.__file__}")
from biomodelml.datasets import SiameseEvolutionDataset


def evaluate_on_test_set(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set (only used after training is complete).

    Args:
        model: The trained model
        test_loader: DataLoader for the test set
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        dict: Dictionary with test metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating on Test Set (FINAL EVALUATION)")
    print("=" * 60)

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

            # Store predictions and targets for additional metrics
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(dist.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader)

    # Calculate additional metrics
    predictions = torch.tensor(all_predictions)
    targets = torch.tensor(all_targets)

    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

    # Correlation coefficient
    pred_mean = predictions.mean()
    target_mean = targets.mean()
    correlation = (((predictions - pred_mean) * (targets - target_mean)).sum() /
                   (torch.sqrt(((predictions - pred_mean) ** 2).sum()) *
                    torch.sqrt(((targets - target_mean) ** 2).sum()))).item()

    print(f"\nTest Set Results:")
    print(f"  Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Correlation: {correlation:.4f}")
    print("=" * 60)

    return {
        'test_loss': avg_test_loss,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }


def train(args):
    """
    The main training loop for the Siamese Regressor.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Data Generator and Dataset
    print("Initializing dataset...")
    if args.cache_dir:
        full_dataset = SiameseEvolutionDataset(cache_dir=args.cache_dir)
    else:
        print("Warning: No cache directory specified. Generating data on-the-fly.")
        generator = SyntheticEvolutionGenerator(seq_len=args.seq_len)
        full_dataset = SiameseEvolutionDataset(
            generator=generator,
            num_samples=args.num_samples,
            max_len=args.max_len
        )

    # 3. Split dataset into training, validation, and test sets
    # Standard split: 70% train, 15% validation, 15% test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    print(f"Dataset split (total: {total_size}):")
    print(f"  Training set:   {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  Validation set: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"  Test set:       {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    print(f"\nNote: Test set is reserved for final evaluation only.")

    # 4. Create DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,      # Faster GPU transfer
        prefetch_factor=4,    # More aggressive prefetching
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    # Test loader is created but NOT used during training (reserved for final evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # 5. Initialize Model, Loss, and Optimizer
    print(f"Initializing model with {args.backbone} backbone...")
    model = SiameseRegressor(
        backbone=args.backbone,
        freeze_backbone=not args.unfreeze_backbone
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # 6. Training Loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        # Wrap train_loader with tqdm for a progress bar
        for i, ((img1, img2), dist) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]")):
            img1, img2 = img1.to(device), img2.to(device)
            dist = dist.to(device)

            optimizer.zero_grad()

            outputs = model(img1, img2)
            loss = criterion(outputs, dist)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Wrap val_loader with tqdm for a progress bar
            for (img1, img2), dist in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Validation]"):
                img1, img2 = img1.to(device), img2.to(device)
                dist = dist.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, dist)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'siamese_regressor.pth'))
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print("Finished Training.")

    # Evaluate on test set if requested
    if args.evaluate_test:
        print("\nLoading best model for test evaluation...")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'siamese_regressor.pth')))
        test_metrics = evaluate_on_test_set(model, test_loader, criterion, device)

        # Save test metrics
        test_metrics_path = os.path.join(args.save_dir, 'test_metrics.txt')
        with open(test_metrics_path, 'w') as f:
            f.write("Final Test Set Evaluation\n")
            f.write("=" * 40 + "\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
        print(f"\nTest metrics saved to {test_metrics_path}")
    else:
        print("\nSkipping test set evaluation (use --evaluate-test to run it).")
        print("Test set is preserved for later evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Siamese Regressor for evolutionary distance prediction.")
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of synthetic pairs to generate.')
    parser.add_argument('--seq_len', type=int, default=500, help='Length of the sequences to simulate.')
    parser.add_argument('--max_len', type=int, default=550, help='Maximum length for padding image matrices.')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'], help='CNN backbone.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.') #default was 32, reduced to 2 for memory constraints
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--unfreeze-backbone', action='store_true', help='If set, the backbone will be trained as well.')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save the trained model.')
    parser.add_argument('--cache_dir', type=str, default='data/siamese_cache', help='Directory to load the cached dataset from.')
    parser.add_argument('--seq_type', type=str, default='N', choices=['N', 'P'],
                        help='Sequence type: N=nucleotide, P=protein (default: N)')
    parser.add_argument('--evaluate-test', action='store_true', help='If set, evaluate on test set after training completes.')

    args = parser.parse_args()
    train(args)
