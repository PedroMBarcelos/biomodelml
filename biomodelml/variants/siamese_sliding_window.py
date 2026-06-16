from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

from biomodelml.structs import DistanceStruct, ImgDebug, ImgDebugs
from biomodelml.variants.deep_search.feature_extractor import FeatureExtractor
from biomodelml.variants.variant import Variant


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_HEAD_CONV_FILTERS = (16, 32)
DEFAULT_HEAD_DENSE_UNITS = (32, 16)


def build_siamese_head_model(
    input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 1),
    conv_filters: Tuple[int, ...] = DEFAULT_HEAD_CONV_FILTERS,
    dense_units: Tuple[int, ...] = DEFAULT_HEAD_DENSE_UNITS,
    dropout: float = 0.1,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="window_distance_matrix")
    x = inputs

    for filters in conv_filters:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="distance")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="siamese_distance_head")


class SiameseSlidingWindowVariant(Variant):
    name = "Siamese Sliding Window Network"

    def __init__(
        self,
        fasta_file: str = None,
        sequence_type: str = None,
        image_folder: str = "",
        window_size: int = 128,
        stride: Optional[int] = None,
        feature_input_shape: Tuple[int, int, int] = (224, 224, 3),
        top_k: int = 4,
        head_path: Optional[str] = None,
        head_scale: float = 1.0,
        head_conv_filters: Tuple[int, ...] = DEFAULT_HEAD_CONV_FILTERS,
        head_dense_units: Tuple[int, ...] = DEFAULT_HEAD_DENSE_UNITS,
        head_dropout: float = 0.1,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        super().__init__(fasta_file, sequence_type)
        if sequence_type is not None and sequence_type not in [
            self.protein_type,
            self.nucleotide_type,
        ]:
            raise IOError(
                f"Sequence must be a protein or nucleotide and is: {sequence_type}"
            )

        self._sequence_type = sequence_type
        self._image_folder = image_folder
        self._window_size = int(window_size)
        self._stride = int(stride) if stride is not None else max(self._window_size // 2, 1)
        self._top_k = max(int(top_k), 1)
        self._feature_input_shape = feature_input_shape
        self._head_path = head_path
        self._head_scale = float(head_scale)
        self._head_conv_filters = tuple(head_conv_filters)
        self._head_dense_units = tuple(head_dense_units)
        self._head_dropout = float(head_dropout)
        self._feature_extractor = feature_extractor or FeatureExtractor(feature_input_shape)
        self._image_index: Dict[str, str] = {}
        self._window_cache: Dict[str, Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]] = {}
        self._head_model: Optional[tf.keras.Model] = None

        if self._head_path:
            self._head_model = tf.keras.models.load_model(self._head_path, compile=False)
            self._load_head_config()

    def _load_head_config(self) -> None:
        if not self._head_path:
            return

        config_path = Path(self._head_path).with_suffix(".json")
        if not config_path.exists():
            return

        with open(config_path, "r") as handle:
            config = json.load(handle)

        self._head_scale = float(config.get("distance_scale", self._head_scale))

    def _load_image_index(self) -> Dict[str, str]:
        if self._image_index:
            return self._image_index

        if not self._image_folder:
            raise IOError("An image folder is required for SiameseSlidingWindowVariant")

        index: Dict[str, str] = {}
        for path in sorted(Path(self._image_folder).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem, str(path))

        if not index:
            raise IOError(f"No images found under {self._image_folder}")

        self._image_index = index
        return index

    def _resolve_names(self) -> List[str]:
        if getattr(self, "_names", None):
            return list(self._names)
        names = sorted(self._load_image_index().keys())
        self._names = names
        return names

    def _read_image(self, img_path: str) -> np.ndarray:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not load image: {img_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _start_positions(self, length: int) -> List[int]:
        if length <= self._window_size:
            return [0]

        starts = list(range(0, length - self._window_size + 1, self._stride))
        last_start = length - self._window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return sorted(set(starts))

    def _window_specs(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        specs: List[Tuple[int, int, int, int]] = []
        for top in self._start_positions(height):
            for left in self._start_positions(width):
                bottom = min(top + self._window_size, height)
                right = min(left + self._window_size, width)
                specs.append((top, left, bottom, right))
        return specs

    def _pad_window(self, window: np.ndarray) -> np.ndarray:
        height, width = window.shape[:2]
        if window.ndim == 2:
            window = cv2.cvtColor(window, cv2.COLOR_GRAY2RGB)

        pad_h = max(0, self._window_size - height)
        pad_w = max(0, self._window_size - width)
        if pad_h == 0 and pad_w == 0:
            return window

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(
            window,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )

    def _extract_windows(
        self, img_name: str
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        if img_name in self._window_cache:
            return self._window_cache[img_name]

        img_path = self._load_image_index()[img_name]
        image = self._read_image(img_path)
        specs = self._window_specs(image)
        windows = [
            self._pad_window(image[top:bottom, left:right])
            for top, left, bottom, right in specs
        ]
        self._window_cache[img_name] = (windows, specs)
        return windows, specs

    def _encode_windows(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        features = [self._feature_extractor.extract(window) for window in windows]
        return np.stack(features, axis=0)

    def _window_distance_matrix(self, img_name1: str, img_name2: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        windows1, _ = self._extract_windows(img_name1)
        windows2, _ = self._extract_windows(img_name2)
        return self._window_distance_matrix_from_windows(windows1, windows2)

    def _window_distance_matrix_from_windows(
        self,
        windows1: Sequence[np.ndarray],
        windows2: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        features1 = self._encode_windows(windows1)
        features2 = self._encode_windows(windows2)

        similarities = np.clip(features1 @ features2.T, -1.0, 1.0)
        distances = 1.0 - similarities

        best_matches = [
            (idx, int(np.argmin(distances[idx])))
            for idx in range(distances.shape[0])
        ]
        return distances, best_matches

    def _window_distance_matrix_from_arrays(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        windows1 = [
            self._pad_window(image1[top:bottom, left:right])
            for top, left, bottom, right in self._window_specs(image1)
        ]
        windows2 = [
            self._pad_window(image2[top:bottom, left:right])
            for top, left, bottom, right in self._window_specs(image2)
        ]
        return self._window_distance_matrix_from_windows(windows1, windows2)

    def _aggregate_distance(self, distances: np.ndarray) -> float:
        row_mins = np.min(distances, axis=1)
        col_mins = np.min(distances, axis=0)

        row_top = np.sort(row_mins)[: min(self._top_k, row_mins.shape[0])]
        col_top = np.sort(col_mins)[: min(self._top_k, col_mins.shape[0])]

        return float((np.mean(row_top) + np.mean(col_top)) / 2.0)

    def _predict_distance(self, distances: np.ndarray) -> float:
        if self._head_model is None:
            return self._aggregate_distance(distances)

        batch = np.expand_dims(np.expand_dims(distances.astype(np.float32), axis=0), axis=-1)
        prediction = self._head_model.predict(batch, verbose=0)
        return float(prediction.reshape(-1)[0] * self._head_scale)

    def _compare(self, img_name1: str, img_name2: str) -> Tuple[float, List[ImgDebug]]:
        distances, best_matches = self._window_distance_matrix(img_name1, img_name2)
        _, specs1 = self._extract_windows(img_name1)
        _, specs2 = self._extract_windows(img_name2)

        score = self._predict_distance(distances)
        debug_rows: List[ImgDebug] = []
        for idx1, idx2 in best_matches:
            top1, left1, bottom1, right1 = specs1[idx1]
            top2, left2, bottom2, right2 = specs2[idx2]
            debug_rows.append(
                ImgDebug(
                    score=f"{float(distances[idx1, idx2]):.6f}",
                    start_col=str(left1),
                    start_line=str(top1),
                    stop_col=str(right1),
                    stop_line=str(bottom1),
                    max_size=f"match:{left2},{top2},{right2},{bottom2}",
                )
            )

        return score, debug_rows

    def predict_from_arrays(self, image1: np.ndarray, image2: np.ndarray) -> float:
        distances, _ = self._window_distance_matrix_from_arrays(image1, image2)
        return self._predict_distance(distances)

    @property
    def head_is_trained(self) -> bool:
        return self._head_model is not None

    def build_matrix(self) -> DistanceStruct:
        names = self._resolve_names()
        missing = set(names).difference(set(self._load_image_index().keys()))
        if missing:
            raise IOError(f"Sequences without image created: {missing}")

        matrix = np.zeros((len(names), len(names)), dtype=np.float64)
        img_debugs: List[ImgDebugs] = []

        for i, name1 in enumerate(names):
            for j in range(i, len(names)):
                name2 = names[j]
                score, debugs = self._compare(name1, name2)
                matrix[i, j] = matrix[j, i] = score
                if debugs:
                    img_debugs.append(ImgDebugs(name1, name2, debugs))

        return DistanceStruct(names=names, matrix=matrix, img_debugs=img_debugs or None)
