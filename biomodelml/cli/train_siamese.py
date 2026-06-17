#!/usr/bin/env python
"""Train the Siamese sliding-window distance head."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from biomodelml.data_generation.training_dataset import TrainingDataset
from biomodelml.variants.siamese_sliding_window import (
    SiameseSlidingWindowVariant,
    build_siamese_head_model,
)


@dataclass(frozen=True)
class PairExample:
    left_index: int
    right_index: int
    label: float
    group_id: str


class SiamesePairSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset: TrainingDataset,
        pairs: Sequence[PairExample],
        variant: SiameseSlidingWindowVariant,
        batch_size: int,
        label_scale: float,
        shuffle: bool = True,
        **kwargs
    ):
        self.dataset = dataset
        self.pairs = list(pairs)
        self.variant = variant
        self.batch_size = max(int(batch_size), 1)
        self.label_scale = max(float(label_scale), 1e-8)
        self.shuffle = shuffle
        self.indices = list(range(len(self.pairs)))
        self._matrix_cache = {} # <-- ADICIONE ESTA LINHA PARA O CACHE
        self.on_epoch_end()

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.pairs) / self.batch_size)))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            random.shuffle(self.indices)
    '''
    def _load_matrix(self, left_index: int, right_index: int) -> np.ndarray:
        left_sample = self.dataset[left_index]
        right_sample = self.dataset[right_index]
        distances, _ = self.variant._window_distance_matrix_from_arrays(
            left_sample.image_array,
            right_sample.image_array,
        )
        return distances.astype(np.float32)
    '''
    def _load_matrix(self, left_index: int, right_index: int) -> np.ndarray:
        # Cria uma chave única para o par
        cache_key = (left_index, right_index)
        
        # Se o par já foi fatiado antes, devolve direto da memória RAM
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]
            
        # Se não estiver no cache (Época 1), faz o cálculo pesado original
        left_sample = self.dataset[left_index]
        right_sample = self.dataset[right_index]
        distances, _ = self.variant._window_distance_matrix_from_arrays(
            left_sample.image_array,
            right_sample.image_array,
        )
        
        final_matrix = distances.astype(np.float32)
        
        # Salva no cache para as próximas épocas
        self._matrix_cache[cache_key] = final_matrix
        return final_matrix
    
    @staticmethod
    def _pad_matrices(matrices: List[np.ndarray]) -> np.ndarray:
        max_height = max(matrix.shape[0] for matrix in matrices)
        max_width = max(matrix.shape[1] for matrix in matrices)
        batch = np.zeros((len(matrices), max_height, max_width, 1), dtype=np.float32)
        for idx, matrix in enumerate(matrices):
            height, width = matrix.shape[:2]
            batch[idx, :height, :width, 0] = matrix
        return batch

    def __getitem__(self, batch_index: int):
        batch_slice = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch_pairs = [self.pairs[index] for index in batch_slice]

        matrices = [self._load_matrix(pair.left_index, pair.right_index) for pair in batch_pairs]
        labels = np.asarray(
            [pair.label / self.label_scale for pair in batch_pairs], dtype=np.float32
        )
        x = self._pad_matrices(matrices)
        y = labels.reshape(-1, 1)
        return x, y


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value")
    return tuple(int(part) for part in parts)


def _collect_pair_examples(dataset: TrainingDataset) -> List[PairExample]:
    manifest_images = dataset.manifest.get("images", [])
    group_map: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for index, image_data in enumerate(manifest_images):
        tree_distances_path = image_data.get("tree_distances_path")
        sequence_name = image_data.get("sequence_name")
        if tree_distances_path and sequence_name:
            group_map[tree_distances_path].append((index, sequence_name))

    pair_examples: List[PairExample] = []
    for tree_distances_path, members in group_map.items():
        distance_frame = pd.read_csv(tree_distances_path, index_col=0)
        for (left_index, left_name), (right_index, right_name) in combinations(members, 2):
            if left_name not in distance_frame.index or right_name not in distance_frame.columns:
                continue
            label = float(distance_frame.loc[left_name, right_name])
            if np.isfinite(label):
                pair_examples.append(
                    PairExample(
                        left_index=left_index,
                        right_index=right_index,
                        label=label,
                        group_id=tree_distances_path,
                    )
                )

    return pair_examples


def _split_examples(
    pairs: Sequence[PairExample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[PairExample], List[PairExample], List[PairExample]]:
    if not pairs:
        return [], [], []

    rng = random.Random(seed)
    group_map: Dict[str, List[PairExample]] = defaultdict(list)
    for pair in pairs:
        group_map[pair.group_id].append(pair)

    groups = list(group_map.keys())
    rng.shuffle(groups)

    if len(groups) < 3:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, train_end + int(total * val_ratio)) if total > 2 else total
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        if not val and test:
            val = test[:1]
            test = test[1:]
        if not test and val:
            test = val[-1:]
            val = val[:-1]
        return train, val, test

    train_cut = max(1, int(len(groups) * train_ratio))
    val_cut = train_cut + max(1, int(len(groups) * val_ratio))
    train_groups = groups[:train_cut]
    val_groups = groups[train_cut:val_cut]
    test_groups = groups[val_cut:]

    train = [pair for group in train_groups for pair in group_map[group]]
    val = [pair for group in val_groups for pair in group_map[group]]
    test = [pair for group in test_groups for pair in group_map[group]]
    return train, val, test


def _build_variant(args: argparse.Namespace) -> SiameseSlidingWindowVariant:
    return SiameseSlidingWindowVariant(
        fasta_file=None,
        sequence_type=args.sequence_type,
        image_folder="",
        window_size=args.window_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_input_shape=tuple(args.feature_input_shape),
        head_conv_filters=_parse_int_tuple(args.conv_filters),
        head_dense_units=_parse_int_tuple(args.dense_units),
        head_dropout=args.dropout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Siamese sliding-window distance head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  biomodelml-train-siamese images_output/ N models/siamese/
  biomodelml-train-siamese images_output/ P models/siamese/ --epochs 8 --batch-size 2
        """,
    )

    parser.add_argument("dataset_root", help="Root directory containing metadata/image_manifest.json")
    parser.add_argument("sequence_type", choices=["N", "P"], help="Sequence type used to generate the matrices")
    parser.add_argument("output_dir", help="Directory where the trained head will be saved")
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k aggregation factor")
    parser.add_argument("--feature-input-shape", type=int, nargs=3, default=[224, 224, 3], help="Feature extractor input shape")
    parser.add_argument("--conv-filters", type=str, default="16,32", help="Comma-separated convolution filters for the head")
    parser.add_argument("--dense-units", type=str, default="32,16", help="Comma-separated dense units for the head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the head")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on the number of pair examples")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset = TrainingDataset(args.dataset_root, lazy_load=True)
    pairs = _collect_pair_examples(dataset)
    if not pairs:
        raise RuntimeError("No pairwise training examples found in the manifest")

    if args.max_pairs is not None:
        rng = random.Random(args.seed)
        pairs = list(pairs)
        rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs]

    train_pairs, val_pairs, test_pairs = _split_examples(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if not train_pairs:
        raise RuntimeError("Training split is empty")

    label_scale = max(pair.label for pair in pairs)
    if label_scale <= 0:
        label_scale = 1.0

    variant = _build_variant(args)
    model = build_siamese_head_model(
        input_shape=(None, None, 1),
        conv_filters=variant._head_conv_filters,
        dense_units=variant._head_dense_units,
        dropout=variant._head_dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    train_sequence = SiamesePairSequence(
        dataset=dataset,
        pairs=train_pairs,
        variant=variant,
        batch_size=args.batch_size,
        label_scale=label_scale,
        shuffle=True,
    )
    val_sequence = (
        SiamesePairSequence(
            dataset=dataset,
            pairs=val_pairs,
            variant=variant,
            batch_size=args.batch_size,
            label_scale=label_scale,
            shuffle=False,
        )
        if val_pairs
        else None
    )
    def train_gen():
        for i in range(len(train_sequence)):
            yield train_sequence[i]

    # Criamos o dataset garantindo assinaturas flexíveis (None) para as matrizes
    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32), # [Batch, Altura, Largura, Canal]
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)              # [Batch, Label]
        )
    )
    # Prefetch ativa o multi-threading de CPU em segundo plano
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Gerador para o Dataset de Validação (se existir)
    if val_sequence is not None:
        def val_gen():
            for i in range(len(val_sequence)):
                yield val_sequence[i]
        
        val_dataset = tf.data.Dataset.from_generator(
            val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        val_dataset = None

    # ====================================================================
    # CONFIGURAÇÃO DO FIT E SALVAMENTO
    # ====================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    head_path = Path(args.output_dir) / "siamese_head.keras"
    config_path = Path(args.output_dir) / "siamese_head.json"

    fit_kwargs = {"epochs": args.epochs, "verbose": 1}

    # Passamos o dataset do tf.data em vez da Sequence crua
    if val_dataset is not None:
        fit_kwargs["validation_data"] = val_dataset

    model.fit(train_dataset, **fit_kwargs)
    model.save(head_path)
    # ====================================================================

    config = {
        "distance_scale": float(label_scale),
        "window_size": args.window_size,
        "stride": args.stride,
        "top_k": args.top_k,
        "feature_input_shape": list(args.feature_input_shape),
        "conv_filters": list(variant._head_conv_filters),
        "dense_units": list(variant._head_dense_units),
        "dropout": variant._head_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "sequence_type": args.sequence_type,
    }
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved trained head to {head_path}")
    print(f"Saved training config to {config_path}")
'''
    os.makedirs(args.output_dir, exist_ok=True)
    head_path = Path(args.output_dir) / "siamese_head.keras"
    config_path = Path(args.output_dir) / "siamese_head.json"

    fit_kwargs = {"epochs": args.epochs, "verbose": 1}

    if val_sequence is not None:
        fit_kwargs["validation_data"] = val_sequence

    model.fit(train_sequence, **fit_kwargs)
    model.save(head_path)

    config = {
        "distance_scale": float(label_scale),
        "window_size": args.window_size,
        "stride": args.stride,
        "top_k": args.top_k,
        "feature_input_shape": list(args.feature_input_shape),
        "conv_filters": list(variant._head_conv_filters),
        "dense_units": list(variant._head_dense_units),
        "dropout": variant._head_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "sequence_type": args.sequence_type,
    }
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved trained head to {head_path}")
    print(f"Saved training config to {config_path}")
'''

if __name__ == "__main__":
    main()