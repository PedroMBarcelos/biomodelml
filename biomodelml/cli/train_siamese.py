import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm

from biomodelml.simulation import SyntheticEvolutionGenerator
from biomodelml.datasets import SiameseEvolutionDataset
from biomodelml.models import SiameseRegressor

def train(args):
    """
    The main training loop for the Siamese Regressor.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Data Generator and Dataset
    print("Initializing data generator and dataset...")
    generator = SyntheticEvolutionGenerator(seq_len=args.seq_len)
    full_dataset = SiameseEvolutionDataset(
        generator=generator,
        num_samples=args.num_samples,
        max_len=args.max_len
    )

    # 3. Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

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

    args = parser.parse_args()
    train(args)
