import tensorflow
from typing import List, Tuple
from tensorflow import Tensor
from biomodelml.variants.ssim_base import SSIMVariant
from biomodelml.structs import ImgDebugs


class ResizedSSIMVariant(SSIMVariant):

    name = "Resized Structural Similarity Index Measure"

    def _upscale_images(self, image: Tensor, other: Tensor) -> Tuple[Tensor]:
        max_x = image.shape[1] if image.shape[1] > other.shape[1] else other.shape[1]
        max_y = image.shape[2] if image.shape[2] > other.shape[2] else other.shape[2]
        return (
            tensorflow.image.resize(
                image, (max_x, max_y), tensorflow.image.ResizeMethod.BICUBIC),
            tensorflow.image.resize(
                other, (max_x, max_y), tensorflow.image.ResizeMethod.BICUBIC)
        )

    def _compare(self, img1: str, img2: str) -> Tuple[float, List[ImgDebugs]]:
        img, other = self._upscale_images(
            self._read_image(img1),
            self._read_image(img2)
        )
        return self._call_alg(img, other), []
