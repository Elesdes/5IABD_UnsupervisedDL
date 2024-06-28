from typing import Tuple
from src.config.path_config import path_config
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def limited_os_walk(data_dir, max_folders):
    folders_explored = 0
    for root, dirs, files in os.walk(data_dir):
        if folders_explored >= max_folders:
            break
        yield root, dirs, files
        folders_explored += 1


def load_custom_dataset(
    data_dir: str = path_config.dataset,
    img_size: Tuple[int, int] | None = (256, 256),
    num_classes: int = 5,
) -> Tuple[Tuple[np.array, np.array], Tuple[None, None]]:
    image_list = []
    target_list = []
    target = 0

    for root, _, files in tqdm(limited_os_walk(data_dir, num_classes)):
        for file in files:
            target_list.append(target)
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert("L")
                    img = img.resize(img_size, Image.LANCZOS)
                    img_array = np.array(img)
                    image_list.append(img_array)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
        target += 1

    return (np.array(image_list), np.array(target_list)), (None, None)
