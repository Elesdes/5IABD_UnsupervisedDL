import random
import numpy as np
from tqdm import tqdm
import pickle
from src.utils.file_utils import avoid_overwrite
from src.config.path_config import path_config
import os


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300):
        self.model_name = "k_means"
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    @classmethod
    def __euclidean(cls, point, data):
        return np.sqrt(np.sum((point - data) ** 2, axis=1))

    def fit(self, X_train):
        self.centroids = [random.choice(X_train)]
        for _ in tqdm(range(self.n_clusters - 1)):
            dists = np.sum(
                [self.__euclidean(centroid, X_train) for centroid in self.centroids],
                axis=0,
            )
            dists /= np.sum(dists)
            (new_centroid_idx,) = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        iteration = 0
        prev_centroids = None
        while (
            np.not_equal(self.centroids, prev_centroids).any()
            and iteration < self.max_iter
        ):
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in tqdm(X_train):
                dists = self.__euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            prev_centroids = self.centroids
            self.centroids = [
                np.mean(cluster, axis=0) for cluster in sorted_points if cluster
            ]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

        model_path = avoid_overwrite(
            filepath=os.path.join(path_config.models, self.model_name, "model.pkl")
        )
        self.__save(model_path)

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = self.__euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

    def compress(self, X):
        _, centroid_idxs = self.evaluate(X)
        return np.array(centroid_idxs)

    def decompress(self, compressed_data):
        return np.array([self.centroids[idx] for idx in compressed_data])

    def generate(self, n_samples):
        if self.centroids is None:
            raise ValueError("Model must be fitted before generating samples.")

        generated_samples = []
        for _ in range(n_samples):
            # Randomly select a centroid
            centroid = random.choice(self.centroids)
            # Add some noise to create variation
            noise = np.random.normal(0, 0.1, size=centroid.shape)
            sample = centroid + noise
            generated_samples.append(sample)

        return np.array(generated_samples)

    @classmethod
    def __save(cls, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(cls, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)
