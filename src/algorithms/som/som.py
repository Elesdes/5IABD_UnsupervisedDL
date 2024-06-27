import numpy as np
import pickle
from tqdm import tqdm


class SOM:
    def __init__(self, x, y, input_dim, learning_rate=0.5, sigma=None):
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(x, y) / 2.0
        self.weights = np.random.rand(x, y, input_dim)
        self.neigx = np.arange(x)
        self.neigy = np.arange(y)
        self.decay_function = lambda x, t, max_iter: x / (1 + t / max_iter)

    def __neighborhood_function(self, distance, sigma):
        return np.exp(-distance / (2 * sigma**2))

    def __calculate_distance(self, x, y, x2, y2):
        return np.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    def __find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_idx

    def __update_weights(self, sample, bmu_idx, iteration, max_iter):
        learning_rate = self.decay_function(self.learning_rate, iteration, max_iter)
        sigma = self.decay_function(self.sigma, iteration, max_iter)

        for i in range(self.x):
            for j in range(self.y):
                dist = self.__calculate_distance(i, j, bmu_idx[0], bmu_idx[1])
                if dist <= sigma:
                    influence = self.__neighborhood_function(dist, sigma)
                    self.weights[i, j, :] += (
                        influence * learning_rate * (sample - self.weights[i, j, :])
                    )

    def train(self, data, num_iterations):
        for iteration in tqdm(range(num_iterations)):
            for sample in data:
                bmu_idx = self.__find_bmu(sample)
                self.__update_weights(sample, bmu_idx, iteration, num_iterations)

    def map_vects(self, data):
        mapped = np.array([self.__find_bmu(sample) for sample in data])
        return mapped

    def compress(self, file_path):
        som_state = {
            "x": self.x,
            "y": self.y,
            "input_dim": self.input_dim,
            "learning_rate": self.learning_rate,
            "sigma": self.sigma,
            "weights": self.weights,
        }
        with open(file_path, "wb") as f:
            pickle.dump(som_state, f)

    @staticmethod
    def decompress(file_path):
        with open(file_path, "rb") as f:
            som_state = pickle.load(f)
        som = SOM(
            x=som_state["x"],
            y=som_state["y"],
            input_dim=som_state["input_dim"],
            learning_rate=som_state["learning_rate"],
            sigma=som_state["sigma"],
        )
        som.weights = som_state["weights"]
        return som

    def generate(self, n_samples: int = 100):
        indices = np.random.randint(0, self.x * self.y, n_samples)
        return self.weights.reshape(self.x * self.y, self.input_dim)[indices]
