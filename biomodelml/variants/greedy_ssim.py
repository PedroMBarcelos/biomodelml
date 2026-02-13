import tensorflow
from tensorflow import Tensor
from biomodelml.variants.ssim_base import SSIMVariant, MAX_POSSIBLE_SCORE
from biomodelml.structs import ImgMap, ImgDebug

THRESHOLD = 0

class GreedySSIMVariant(SSIMVariant):

    name = "Greedy Sliced Structural Similarity Index Measure"

    def _greedy_find_image_match(
        self,
        small_img: Tensor,
        big_img: Tensor,
        c11: int, c12: int,
        l21: int, l22: int,
        c21: int, c22: int,
        max_score: int, last_line: int, step: int
    ):
        if (c22 > big_img.shape[1]) or (l22 > big_img.shape[0]):
            return max_score, last_line

        score = self._call_alg(
            tensorflow.expand_dims(small_img[:, c11:c12], axis=0),
            tensorflow.expand_dims(big_img[l21:l22, c21:c22], axis=0))
        if score == MAX_POSSIBLE_SCORE:
            return score, l21
        elif score > max_score:
            max_score = score
            last_line = l21
        return self._greedy_find_image_match(
            small_img, big_img, c11, c12, l21+step, l22+step, c21+step, c22+step, max_score, last_line, step)

    def _find_best_col(self, min_img: Tensor, max_img: Tensor, mask_size: int, filter_size: int, step: int) -> ImgMap:
        last_line = 0
        scores = list()
        debugs = list()
        start_score = 0
        for j in range(0, mask_size - filter_size, step):
            window = min(mask_size, j+filter_size)
            last_score, last_line = self._greedy_find_image_match(
                min_img, max_img,
                j, window,
                last_line, mask_size+last_line,
                last_line+j, last_line+window,
                start_score, last_line, step
            )
            if last_score > THRESHOLD:
                scores.append(last_score)
                debugs.append(
                    ImgDebug(str(last_score), str(last_line+j), str(last_line),
                            str(last_line+window), str(mask_size+last_line), str(max_img.shape[1])))
            else:
                last_line = int(debugs[-1].stop_col) if len(debugs) > 0 else 0
        return ImgMap(debugs=debugs, scores=scores)
