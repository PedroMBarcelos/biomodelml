import os
import pandas
import numpy
import tensorflow
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import List, Tuple
from tensorflow import Tensor
from biomodelml.context import RecursionContext
from biomodelml.variants.variant import Variant
from biomodelml.structs import DistanceStruct, ImgMap, ImgDebug, ImgDebugs


MAX_POSSIBLE_SCORE = 1.0
DEFAULT_PARAMS = dict(
    N=dict(
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        filter_size=11,
        max_val=255
    ), P=dict(
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        filter_size=4,
        max_val=255
    ),
)


class SSIMVariant(Variant):

    name = "Structural Similarity Index Measure"

    def __init__(self, fasta_file: str = None, sequence_type: str = None, image_folder: str = "", **alg_params):
        super().__init__(fasta_file, sequence_type)
        self._image_folder = image_folder
        self._alg_params = DEFAULT_PARAMS[sequence_type].copy()
        self._alg_params.update(alg_params)
        self.filter_size = self._alg_params["filter_size"]
        self._executor = ThreadPoolExecutor(max_workers=cpu_count())

    def _call_alg(self, image: Tensor, other: Tensor) -> float:
        return tensorflow.image.ssim(
            image,
            other,
            **self._alg_params)[0].numpy()

    def _read_image(self, img_name: str) -> Tensor:
        img = tensorflow.expand_dims(
            tensorflow.image.decode_image(
                tensorflow.io.read_file(
                    os.path.join(
                        self._image_folder, img_name)), channels=3), axis=0)
        if tensorflow.math.equal(img[:, :, :, 2], img[:, :, :, 1], img[:, :, :, 0]).numpy().all():
            img = tensorflow.image.rgb_to_grayscale(img)
        return img

    def _find_best_col(self, min_img: Tensor, max_img: Tensor, mask_size: int, filter_size: int, step: int) -> ImgMap:
        raise NotImplementedError(
            "Base class should implement this before use")

    def _match_images(self, image: Tensor, other: Tensor) -> Tuple[float, List[ImgDebug]]:
        if image.shape[1] > other.shape[1]:
            max_img = image[0]
            min_img = other[0]
        else:
            min_img = image[0]
            max_img = other[0]

        mask_size = min_img.shape[1]
        filter_size = self.filter_size
        step = max(filter_size // 2, 1)
        default_args = (mask_size, filter_size, step)
        img_map = self._find_best_col(min_img, max_img, *default_args)
        return numpy.mean(img_map.scores), img_map.debugs

    def _compare(self, img1: str, img2: str) -> Tuple[float, List[ImgDebugs]]:
        img = self._read_image(img1)
        other = self._read_image(img2)
        with RecursionContext():
            result, debugs = self._match_images(img, other)
            img_debugs = [ImgDebugs(img1, img2, debugs)] if debugs else []
        print(f"{img1} and {img2} done with score {result}")
        return result, img_debugs

    def calc_alg(self, img_name1: str, img_name2: str) -> float:
        return self._compare(img_name1, img_name2)[0]

    def build_matrix(self) -> DistanceStruct:
        files = os.listdir(self._image_folder)
        indexes = {".".join(img.split('.')[:-1]): img.split('.')[-1] for img in files}
        diff = set(self._names).difference(set(indexes.keys()))
        if diff:
            raise IOError(f"Sequences without image created: {diff}")
        files = []
        for i in self._names:
            if indexes.get(i):
                files.append(f"{i}.{indexes.get(i)}")
        indexes = self._names
        df = pandas.DataFrame(index=indexes, columns=indexes)
        last_ids = list()
        img_debugs = list()
        for idx, img1 in enumerate(files):
            results = list()
            idx1 = indexes[idx]
            futures = [self._executor.submit(
                self._compare, img1, img2) for img2 in files[idx:]]
            for future in futures:
                r, d = future.result()
                results.append(r)
                img_debugs += d
            if last_ids:
                df.loc[idx1, indexes[idx:]] = results
                df.loc[idx1, last_ids] = df.loc[last_ids, idx1]
            else:
                df.loc[idx1, :] = results
            last_ids.append(idx1)
        return DistanceStruct(
            names=indexes,
            matrix=1.0-df.to_numpy(numpy.float64),
            img_debugs=img_debugs)
