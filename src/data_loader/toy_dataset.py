from typing import Tuple
import numpy as np
from sklearn.datasets import make_blobs


def load_toy_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_clusters: int | None = None,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
    )
