import tensorflow
from tensorflow import Tensor
from typing import Tuple, List
from biomodelml.structs import ImgDebug
from biomodelml.variants.ssim_multiscale_base import SSIMMultiScaleVariant


class WindowedSSIMMultiScaleVariant(SSIMMultiScaleVariant):

    name = "Windowed MultiScale Structural Similarity Index Measure"

    def _dynamic_find_image_match(
        self,
        small_img: Tensor,
        big_img: Tensor,
        s1: int, s2: int,
        step: int, window: int
    ) -> Tuple[float, List[ImgDebug]]:
        max_score = 0
        debugs = list()
        for i in range(0, big_img.shape[1]-window+1, step):
            score = self._call_alg(
                tensorflow.expand_dims(small_img, axis=0),
                tensorflow.expand_dims(big_img[s1+i:s2+i, s1+i:s2+i], axis=0))
            max_score = max((score, max_score))
            debugs.append(
                ImgDebug(str(score), str(s1+i), str(s1+i), str(s2+i), str(s2+i), str(big_img.shape[1])))
        return max_score, debugs

    def _match_images(self, image: Tensor, other: Tensor) -> Tuple[float, List[ImgDebug]]:
        if image.shape[1] > other.shape[1]:
            max_img = image[0]
            min_img = other[0]
        else:
            min_img = image[0]
            max_img = other[0]
        filter_size = min_img.shape[1]
        step = 1
        return self._dynamic_find_image_match(
            min_img, max_img, 0, filter_size, step, filter_size)
