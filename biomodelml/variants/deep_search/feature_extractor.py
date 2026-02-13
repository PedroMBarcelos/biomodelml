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
        if img.shape[0] < self._input_shape[0]:
            diff_shape = self._input_shape[0] - img.shape[0]
            if diff_shape % 2 > 0:
                diff_shape -= 1
                img = cv2.resize(
                    img,
                    dsize=(img.shape[0]+1, img.shape[1]+1),
                    interpolation=cv2.INTER_CUBIC)
            new_image = numpy.zeros(self._input_shape)
            pad = diff_shape // 2
            new_image[pad:-pad, pad:-pad] = img
            return new_image
        else:
            return self._upscale(img)

    def extract(self, img: numpy.ndarray) -> numpy.ndarray:
        img = self._upscale_with_border(img)
        x = numpy.expand_dims(img, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return (feature / numpy.linalg.norm(feature)).reshape((feature.size, 1))

    def get_feature(self, image_data: List[str]):
        self.image_data = image_data 
        features = []
        for img_path in self.image_data:
            try:
                feature = self.extract(img=cv2.imread(img_path))
                features.append(feature)
            except Exception as e:
                raise e
        return features