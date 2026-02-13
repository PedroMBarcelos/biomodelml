import os
import pandas
import numpy
import cv2
from typing import Tuple
from sewar.full_ref import uqi
from multiprocessing.dummy import Pool
from biomodelml.variants.variant import Variant
from biomodelml.structs import DistanceStruct


class UQIVariant(Variant):

    name = "Universal Quality Index"

    def __init__(self, fasta_file: str = None, sequence_type: str = None, image_folder: str = ""):
        super().__init__(fasta_file, sequence_type)
        self._image_folder = image_folder
        self._filter_size = 11 if sequence_type == "N" else 4

    
    def _read_image(self, img_name: str) -> numpy.ndarray:
        return cv2.imread(
            os.path.join(
                self._image_folder, img_name))       

    def _upscale_images(self, image: numpy.ndarray, other: numpy.ndarray) -> Tuple[numpy.ndarray]:
        max_x = image.shape[0] if image.shape[0] > other.shape[0] else other.shape[0]
        max_y = image.shape[1] if image.shape[1] > other.shape[1] else other.shape[1]
        result = (
            cv2.resize(image, dsize=(max_x, max_y), interpolation=cv2.INTER_CUBIC),
            cv2.resize(other, dsize=(max_x, max_y), interpolation=cv2.INTER_CUBIC)
        )
        return result

    def _upscale_with_border(self, image: numpy.ndarray, other: numpy.ndarray) -> Tuple[numpy.ndarray]:
        if image.shape[0] < other.shape[0]:
            min_image = image
            max_image = other
        elif image.shape[0] > other.shape[0]:
            min_image = other
            max_image = image
        else:
            return (image, other)
        diff_shape = max_image.shape[0] - min_image.shape[0]
        if diff_shape % 2 > 0:
            diff_shape += 1
            max_image = cv2.resize(
                max_image,
                dsize=(max_image.shape[0]+1, max_image.shape[1]+1),
                interpolation=cv2.INTER_CUBIC)
        new_image = numpy.zeros(max_image.shape)
        pad = diff_shape // 2
        new_image[pad:-pad, pad:-pad] = min_image
        return (max_image, new_image)

    def calc_alg(self, img_name1: str, img_name2: str) -> float:
        return uqi(
            *self._upscale_images(
                self._read_image(img_name1), self._read_image(img_name2)), self._filter_size)

    def build_matrix(self) -> DistanceStruct:
        files = os.listdir(self._image_folder)
        indexes = {".".join(img.split('.')[:-1]): img.split('.')[-1] for img in files}
        threads = 3
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
        print(f"doing for {len(files)} sequences")
        for idx, img1 in enumerate(files):
            idx1 = img1.split('.')[0]
            with Pool(threads) as pool:
                result = pool.starmap(
                    self.calc_alg,
                    [(img1, img2) for img2 in files[idx:]]
                )
            if last_ids:
                df.loc[idx1, indexes[idx:]] = result
                df.loc[idx1, last_ids] = df.loc[last_ids, idx1]
            else:
                df.loc[idx1, :] = result
            last_ids.append(idx1)
        return DistanceStruct(
            names=indexes, matrix=abs(1.0-df.to_numpy(numpy.float64)))
