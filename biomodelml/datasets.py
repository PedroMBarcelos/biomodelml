import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from Bio.Seq import Seq


from biomodelml.simulation import SyntheticEvolutionGenerator
from biomodelml.matrices import build_matrix

class SiameseEvolutionDataset(Dataset):
    """
    PyTorch Dataset for training the Siamese Regressor.
    This dataset generates pairs of sequences on-the-fly using the
    SyntheticEvolutionGenerator, converts them to image matrices, and
    applies necessary transformations.
    """
    def __init__(self, generator, num_samples=100000, max_len=500, transform=None):
        """
        Args:
            generator (SyntheticEvolutionGenerator): The data generator instance.
            num_samples (int): The total number of samples (pairs) in the dataset.
            max_len (int): The maximum sequence length for padding matrices.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.generator = generator
        self.num_samples = num_samples
        self.max_len = max_len
        self.transform = transform or self._get_default_transform()

    def _get_default_transform(self):
        """
        Returns a default transformation pipeline for the image matrices.
        This includes converting to a tensor and applying normalization.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates and returns one sample from the dataset.
        A sample consists of a pair of image matrices and their evolutionary distance.
        """
        # 1. Generate a synthetic evolution pair
        parent_seq, mutated_seq, distance = self.generator.generate_evolution_pair()

        # 2. Convert sequences to an RGB image matrix
        # The build_matrix function returns a NumPy array of shape (H, W, 3)
        # We need to pad it to a fixed size (max_len x max_len)
        image_matrix = build_matrix(parent_seq.seq, mutated_seq.seq, self.max_len, seq_type='N')

        # 3. Pad the matrix to the max_len
        h, w, c = image_matrix.shape
        padded_matrix = np.zeros((self.max_len, self.max_len, c), dtype=np.uint8)
        padded_matrix[:h, :w, :] = image_matrix

        # 4. Create two "views" of the data for the Siamese network.
        # In our case, the matrix itself represents the comparison, so we
        # create two dummy inputs. The real input will be the combined matrix.
        # However, the spec implies two separate inputs. We will pass the same
        # padded matrix as both inputs, as the matrix represents the *relationship*
        # between the two sequences. A better approach is to have two separate matrices
        # vs a reference, but we will follow the spec's logic of comparing I1 and I2.
        # For our case, I1 is parent vs mutated, and I2 is a placeholder.
        # Let's correct this: The network expects two images, one for each sequence.
        # Each image should be a self-comparison matrix.
        img1 = build_matrix(parent_seq.seq, parent_seq.seq, self.max_len, seq_type='N')
        img2 = build_matrix(mutated_seq.seq, mutated_seq.seq, self.max_len, seq_type='N')

        # Pad them
        h1, w1, c1 = img1.shape
        padded_img1 = np.zeros((self.max_len, self.max_len, c1), dtype=np.uint8)
        padded_img1[:h1, :w1, :] = img1

        h2, w2, c2 = img2.shape
        padded_img2 = np.zeros((self.max_len, self.max_len, c2), dtype=np.uint8)
        padded_img2[:h2, :w2, :] = img2


        # 5. Apply transformations
        if self.transform:
            img1_tensor = self.transform(padded_img1)
            img2_tensor = self.transform(padded_img2)

        # 6. Convert distance to a tensor
        distance_tensor = torch.tensor([distance], dtype=torch.float32)

        return (img1_tensor, img2_tensor), distance_tensor

if __name__ == '__main__':
    # Example of how to use the Dataset
    print("Initializing SyntheticEvolutionGenerator...")
    gen = SyntheticEvolutionGenerator(seq_len=100)

    print("Initializing SiameseEvolutionDataset...")
    dataset = SiameseEvolutionDataset(generator=gen, num_samples=10, max_len=150)

    print(f"Dataset length: {len(dataset)}")

    # Get one sample
    (img1, img2), dist = dataset[0]
    print(f"Sample 0:")
    print(f"  Image 1 Tensor Shape: {img1.shape}")
    print(f"  Image 2 Tensor Shape: {img2.shape}")
    print(f"  Distance Tensor: {dist}")
    print(f"  Distance Value: {dist.item():.4f}")
