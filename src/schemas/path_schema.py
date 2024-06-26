from dataclasses import dataclass
import os


@dataclass
class PathSchema:
    root: str = os.path.abspath(".")

    # Data
    data: str = os.path.join(root, "data")
    models: str = os.path.join(data, "models")
    dataset: str = os.path.join(data, "dataset")

    # Algorithms
    algorithms: str = os.path.join(root, "notebooks")

    def __create_file_tree(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)

        for _, dirs, _ in os.walk(self.algorithms):
            for dir in dirs:
                model_path = os.path.join(self.models, dir)
                os.makedirs(model_path, exist_ok=True)

    def __post_init__(self):
        self.__create_file_tree()
