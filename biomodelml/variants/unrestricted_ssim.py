import tensorflow
import numpy
from tensorflow import Tensor
from typing import Tuple, List
from biomodelml.variants.ssim_base import SSIMVariant
from biomodelml.structs import ImgDebug

THRESHOLD = 0

class UnrestrictedSSIMVariant(SSIMVariant):

    name = "Unrestricted Sliced Structural Similarity Index Measure"

    def __init__(self, fasta_file: str = None, sequence_type: str = None, image_folder: str = "", **alg_params):
        alg_params["return_index_map"] = True
        super().__init__(fasta_file, sequence_type, image_folder, **alg_params)

    def _dynamic_find_image_match(
        self,
        small_img: Tensor,
        big_img: Tensor,
        s1: int, s2: int,
        step: int, window: int
    ) -> Tuple[float, List[ImgDebug]]:
        debugs = list()
        score_list = [0 for i in range(window)]
        for i in range(0, big_img.shape[1]-window+1, step):
            scores = self._call_alg(
                tensorflow.expand_dims(small_img, axis=0),
                tensorflow.expand_dims(big_img[s1+i:s2+i, s1+i:s2+i], axis=0))
            max_scores = tensorflow.math.reduce_mean(scores, axis=0).numpy()
            for k, s in enumerate(max_scores):
                if score_list[k] < s:
                    score_list[k] = s
                    debugs.append(
                        ImgDebug(str(s), str(s1+i+k), str(s1+i+k), str(s2+i), str(s2+i), str(big_img.shape[1])))
        score_array = numpy.array(score_list)
        idx_without_zero_less = numpy.where(score_array > THRESHOLD)
        return numpy.mean(score_array[idx_without_zero_less]), debugs

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
