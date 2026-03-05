import os
import glob
import pandas
import numpy
import hashlib
from typing import List
from multiprocessing import cpu_count
from biomodelml.variants.deep_search.feature_extractor import FeatureExtractor
from annoy import AnnoyIndex


class Indexer:
    distance_type = "euclidean"

    def __init__(self, image_folder: str, seq_names: str, feature_extractor: FeatureExtractor):
        self.image_list = []
        self._index = None
        self._cores = cpu_count()
        self._idx_path = os.path.join(image_folder, hashlib.md5(".".join(sorted(seq_names)).encode()).hexdigest()+".ann")
        for path in glob.iglob(f'{image_folder}/**/*.png', recursive=True):
            if os.path.basename(".".join(path.split(".")[:-1])) in seq_names:
                self.image_list.append(path)
        self._feature_extractor = feature_extractor

    def _feature_extraction(self) -> numpy.ndarray:
        """Extract features from all images with error handling"""
        if not self.image_list:
            raise ValueError(f"No images found in the specified folder")
        
        image_data = pandas.DataFrame()
        image_data['images_paths'] = self.image_list
        
        try:
            f_data = self._feature_extractor.get_feature(self.image_list)
            image_data['features'] = f_data
            image_data = image_data.dropna().reset_index(drop=True)
            
            if len(image_data) == 0:
                raise ValueError("No valid features extracted from images")
            
            return image_data
        except Exception as e:
            print(f"  ✗ Feature extraction failed: {e}")
            raise
    
    def _construct(self, data: numpy.ndarray):
        """Construct Annoy index from extracted features"""
        self._index = AnnoyIndex(self._feature_extractor.item_size, self.distance_type)
        
        for i, v in zip(data.index, data['features']):
            # Ensure feature is 1D numpy array
            if isinstance(v, numpy.ndarray):
                v = v.flatten()
            else:
                v = numpy.array(v).flatten()
            
            # Verify feature dimension matches
            if len(v) != self._feature_extractor.item_size:
                raise ValueError(
                    f"Feature size mismatch: expected {self._feature_extractor.item_size}, "
                    f"got {len(v)} for image at index {i}"
                )
            
            self._index.add_item(i, v)
        
        trees = max(10, len(data["features"]))  # At least 10 trees for stability
        self._index.build(trees, n_jobs=self._cores)
        self._index.save(self._idx_path)

    def load_or_build(self):
        print(f"index path: {self._idx_path}")
        if os.path.exists(self._idx_path):
            print("loading index...")
            self._index = AnnoyIndex(self._feature_extractor.item_size, self.distance_type)
            self._index.load(self._idx_path)
        else:
            print("building index...")
            self.build()

    def build(self) -> numpy.ndarray:
        data = self._feature_extraction()
        self._construct(data)
        return data
    
    def search_by_item(self, item: int) -> List[float]:
        if not self._index:
            raise Exception("Run build first")
        return self._index.get_nns_by_item(
            item, -1, search_k=-1, include_distances=True)

    def get_distance(self, item1: int, item2: int) -> float:
        if not self._index:
            raise Exception("Run build first")
        return self._index.get_distance(item1, item2)