import numpy
import cv2
from typing import Tuple, List
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


class FeatureExtractor:
    def __init__(self, input_shape: Tuple[int]):
        self._input_shape = input_shape
        self.model = VGG16(
            include_top=False, input_shape=self._input_shape, weights='imagenet')

    @property
    def item_size(self):
        return numpy.prod(self.model.output.shape[1:])

    def _upscale(self, img: numpy.ndarray) -> numpy.ndarray:
        return cv2.resize(
            img,
            dsize=(self._input_shape[0], self._input_shape[1]),
            interpolation=cv2.INTER_CUBIC)

    def _upscale_with_border(self, img: numpy.ndarray) -> numpy.ndarray:
        """Upscale image with reflection padding (better than black borders)"""
        target_h, target_w = self._input_shape[:2]
        h, w = img.shape[:2]
        
        # If already large enough, just resize
        if h >= target_h and w >= target_w:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate padding needed
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        # Ensure image has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] != 3:
            img = img[:, :, :3]
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Use reflection padding instead of black borders
        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_REFLECT_101
        )
        
        # Ensure exact target size
        if padded.shape[:2] != (target_h, target_w):
            padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        return padded

    def extract(self, img: numpy.ndarray) -> numpy.ndarray:
        """Extract and normalize feature vector from image"""
        img = self._upscale_with_border(img)
        x = numpy.expand_dims(img, axis=0)
        x = preprocess_input(x)
        
        # Extract features (verbose=0 to reduce logs)
        feature = self.model.predict(x, verbose=0)[0]
        
        # Flatten to 1D array (CRITICAL: Annoy requires 1D arrays)
        feature = feature.flatten()
        
        # L2 normalization
        norm = numpy.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        return feature

    def get_feature(self, image_data: List[str]):
        """Extract features from list of image paths"""
        self.image_data = image_data 
        features = []
        for img_path in self.image_data:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not load image: {img_path}")
                feature = self.extract(img=img)
                features.append(feature)
            except Exception as e:
                print(f"  ⚠️  Error processing {img_path}: {e}")
                raise e
        return features