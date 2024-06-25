from dataclasses import dataclass
import os


@dataclass
class PathSchema:
    root: str = os.path.abspath(".")

    # Data
    data: str = os.path.join(root, "data")
    models: str = os.path.join(data, "models")

    # Algorithms
    algorithms: str = os.path.join(root, "notebooks", "algorithms")

    def __create_models_dir(self):
        for _, _, files in os.walk(self.algorithms):
            for file in files:
                if file.endswith(".ipynb"):
                    model_name = os.path.splitext(file)[0].title()
                    model_path = os.path.join(self.models, model_name)
                    os.makedirs(model_path, exist_ok=True)

    def __post_init__(self):
        os.makedirs(PathSchema.models, exist_ok=True)
        self.__create_models_dir()
