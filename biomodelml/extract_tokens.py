# extract_tokens.py
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

from biomodelml.data_generation.training_dataset import TrainingDataset
from biomodelml.variants.siamese_sliding_window import SiameseSlidingWindowVariant

def is_window_informative(window: np.ndarray, threshold: float) -> bool:
    if np.std(window) < 1e-3: return False
    return np.mean(window <= 1e-2) < threshold

def build_encoder(embedding_dim=256):
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(embedding_dim, activation=None)(x)
    return models.Model(inputs, outputs, name="Token_Encoder")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("sequence_type", choices=["N", "P"])
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=180)
    parser.add_argument("--sparsity-threshold", type=float, default=0.85)
    args = parser.parse_args()

    dataset = TrainingDataset(args.dataset_root, lazy_load=True)
    variant = SiameseSlidingWindowVariant(
        image_folder=args.dataset_root, window_size=args.window_size,
        stride=args.stride, top_k=4, sequence_type=args.sequence_type
    )

    out_dir = Path("extracted_tokens") / f"window_{args.window_size}_{args.sequence_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder(embedding_dim=256)
    
    # Coleta índices únicos de imagens no manifesto
    manifest_images = dataset.manifest.get("images", [])
    
    print("-> Extraindo tokens geométricos das imagens exclusivas...")
    for idx in tqdm(range(len(manifest_images)), desc="Processando Imagens"):
        sample = dataset[idx]
        specs = variant._window_specs(sample.image_array)
        
        valid_windows = []
        for top, left, bottom, right in specs:
            win = sample.image_array[top:bottom, left:right]
            padded_win = variant._pad_window(win)
            if is_window_informative(padded_win, args.sparsity_threshold):
                valid_windows.append(padded_win)
                
        if valid_windows:
            windows_batch = np.array(valid_windows, dtype=np.float32)
            # Resize rápido em lote na GPU
            windows_resized = tf.image.resize(windows_batch, [224, 224])
            
            # Mini-batching de 32 para blindar contra OOM
            embeddings = []
            for i in range(0, len(windows_resized), 32):
                chunk = windows_resized[i : i + 32]
                chunk_emb = encoder(chunk, training=False).numpy()
                embeddings.append(chunk_emb)
                
            token_matrix = np.concatenate(embeddings, axis=0)
            np.save(out_dir / f"tokens_{idx}.npy", token_matrix)
        else:
            # Imagem vazia ganha um vetor nulo básico
            np.save(out_dir / f"tokens_{idx}.npy", np.zeros((1, 256), dtype=np.float32))

    print(f"[FIM] Tokens salvos em: {out_dir}")

if __name__ == "__main__":
    main()