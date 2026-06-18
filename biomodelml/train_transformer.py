# train_transformer.py
import os
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

from biomodelml.data_generation.training_dataset import TrainingDataset
from biomodelml.cli.train_siamese import _collect_pair_examples

def build_phylogeny_transformer(embedding_dim=256, num_heads=4):
    tokens_A = layers.Input(shape=(None, embedding_dim), name="seq_A")
    tokens_B = layers.Input(shape=(None, embedding_dim), name="seq_B")
    
    # Mecanismo de Atenção Cruzada Filogenética
    cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
    attn_out = cross_attn(query=tokens_A, value=tokens_B)
    
    x = layers.Add()([tokens_A, attn_out])
    x = layers.LayerNormalization()(x)
    
    # Feed Forward
    ffn = layers.Dense(512, activation="relu")(x)
    ffn = layers.Dense(embedding_dim)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)
    
    # Regressão Final
    glob_pool = layers.GlobalAveragePooling1D()(x)
    dense = layers.Dense(64, activation="relu")(glob_pool)
    outputs = layers.Dense(1, activation="linear")(dense)
    
    return models.Model(inputs=[tokens_A, tokens_B], outputs=outputs)

def token_data_generator(pairs, token_dir, label_scale):
    """Gera amostras dinâmicas par por par para aceitar formatos variantes (N e M)"""
    while True:
        random.shuffle(pairs)
        for pair in pairs:
            try:
                tA = np.load(token_dir / f"tokens_{pair.left_index}.npy")
                tB = np.load(token_dir / f"tokens_{pair.right_index}.npy")
                
                # Transforma em formato batch (1, N, 256) e (1, M, 256)
                x_A = np.expand_dims(tA, axis=0)
                x_B = np.expand_dims(tB, axis=0)
                y = np.array([[pair.label / label_scale]], dtype=np.float32)
                
                yield ({"seq_A": x_A, "seq_B": x_B}, y)
            except FileNotFoundError:
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("sequence_type", choices=["N", "P"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    token_dir = Path("extracted_tokens") / f"window_200_{args.sequence_type}"
    dataset = TrainingDataset(args.dataset_root, lazy_load=True)
    pairs = _collect_pair_examples(dataset)
    
    label_scale = max(pair.label for pair in pairs) if pairs else 1.0

    model = build_phylogeny_transformer(embedding_dim=256)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

    # Criamos o pipeline estável via tf.data
    output_signature = (
        {"seq_A": tf.TensorSpec(shape=(1, None, 256), dtype=tf.float32),
         "seq_B": tf.TensorSpec(shape=(1, None, 256), dtype=tf.float32)},
        tf.TensorSpec(shape=(1, 1), dtype=tf.float32)
    )
    
    train_dataset = tf.data.Dataset.from_generator(
        lambda: token_data_generator(pairs, token_dir, label_scale),
        output_signature=output_signature
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("\n-> Iniciando Treinamento do Transformer Siamês...")
    model.fit(train_dataset, steps_per_epoch=len(pairs), epochs=args.epochs, verbose=1)
    
    model.save("models/transformer_phylogeny.keras")
    print("[SUCESSO] Modelo Transformer salvo!")

if __name__ == "__main__":
    main()