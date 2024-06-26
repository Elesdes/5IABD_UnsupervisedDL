import numpy as np


class SOM:
    def __init__(
        self,
        x,
        y,
        input_len: int,
        learning_rate: float = 0.5,
        radius: int | None = None,
        radius_decay: float = 0.995,
        learning_rate_decay: float = 0.995,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(x, y) / 2
        self.radius_decay = radius_decay
        self.learning_rate_decay = learning_rate_decay
        self.weights = np.random.random((x, y, input_len))

    def train(self, data, num_iterations: int):
        for i in range(num_iterations):
            for vector in data:
                bmu_idx = self._find_bmu(vector)
                self._update_weights(vector, bmu_idx, i, num_iterations)
            self.radius *= self.radius_decay
            self.learning_rate *= self.learning_rate_decay

    def _find_bmu(self, vector):
        bmu_idx = np.argmin(np.linalg.norm(self.weights - vector, axis=-1))
        return np.unravel_index(bmu_idx, (self.x, self.y))

    def _update_weights(
        self, vector, bmu_idx: int, iteration: int, num_iterations: int
    ):
        learning_rate = self.learning_rate * np.exp(-iteration / num_iterations)
        radius = self.radius * np.exp(-iteration / num_iterations)
        for x in range(self.x):
            for y in range(self.y):
                weight = self.weights[x, y]
                distance = np.linalg.norm(np.array([x, y]) - np.array(bmu_idx))
                if distance <= radius:
                    influence = np.exp(-distance / (2 * (radius**2)))
                    self.weights[x, y] += learning_rate * influence * (vector - weight)

    def map_vects(self, data):
        mapped = np.array([self._find_bmu(vector) for vector in data])
        return mapped
