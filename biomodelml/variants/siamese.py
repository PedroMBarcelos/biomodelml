import torch
import numpy as np
import os
from Bio.Seq import Seq

from biomodelml.models import SiameseRegressor
from biomodelml.variants.variant import Variant
from biomodelml.structs import DistanceStruct
from biomodelml.matrices import build_matrix
from torchvision import transforms

class SiameseVariant(Variant):
    """
    A variant that uses a trained Siamese Neural Network to calculate the
    distance between sequences.
    """
    name = "siamese"

    def __init__(self, *args, model_path='models/siamese_regressor.pth', max_len=550, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        """Loads the trained SiameseRegressor model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please train the model first.")
        print(f"Loading model from {model_path} onto {self.device}...")
        # Assuming the model was saved with a resnet50 backbone
        model = SiameseRegressor(backbone='resnet50', pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """Returns the transformation pipeline for input images."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _preprocess_pair(self, seq1, seq2):
        """
        Preprocesses a pair of sequences into padded image tensors.
        """
        # Each image should be a self-comparison matrix.
        img1 = build_matrix(seq1, seq1, self.max_len, seq_type=self.seq_type)
        img2 = build_matrix(seq2, seq2, self.max_len, seq_type=self.seq_type)

        # Pad matrices
        def pad_image(img):
            h, w, c = img.shape
            padded = np.zeros((self.max_len, self.max_len, c), dtype=np.uint8)
            padded[:h, :w, :] = img
            return padded

        padded_img1 = pad_image(img1)
        padded_img2 = pad_image(img2)

        # Apply transforms
        tensor1 = self.transform(padded_img1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(padded_img2).unsqueeze(0).to(self.device)

        return tensor1, tensor2

    def build_matrix(self):
        """
        Calculates the distance matrix for all sequence pairs using the
        Siamese network.
        """
        num_seqs = len(self.seqs)
        dist_matrix = np.zeros((num_seqs, num_seqs))

        print(f"Calculating {self.name} distance matrix for {num_seqs} sequences...")

        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                seq1 = self.seqs[i].seq
                seq2 = self.seqs[j].seq

                # Preprocess the pair and get tensors
                tensor1, tensor2 = self._preprocess_pair(seq1, seq2)

                # Predict distance with the model
                with torch.no_grad():
                    distance = self.model(tensor1, tensor2).item()

                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance # Symmetric matrix

        names = [seq.id for seq in self.seqs]
        return DistanceStruct(
            names=names,
            matrix=dist_matrix
        )
