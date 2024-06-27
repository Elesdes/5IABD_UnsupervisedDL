import os


class PathSchema:
    def __init__(self):
        self.root: str = os.path.abspath(
            os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
        )

        # Data
        self.data: str = os.path.join(self.root, "data")
        self.models: str = os.path.join(self.data, "models")
        self.dataset: str = os.path.join(self.data, "dataset")

        # Algorithms
        self.algorithms: str = os.path.join(self.root, "src", "algorithms")

        self.__create_file_tree()

    def __create_file_tree(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)

        for _, dirs, _ in os.walk(self.algorithms):
            for dir in dirs:
                if "__" not in dir:
                    model_path = os.path.join(self.models, dir)
                    print(model_path)
                    os.makedirs(model_path, exist_ok=True)
