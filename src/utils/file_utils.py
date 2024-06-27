import os


def avoid_overwrite(*, filepath: str):
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    i = 1
    new_filepath = f"{base}_{i}{ext}"

    while os.path.exists(new_filepath):
        i += 1
        new_filepath = f"{base}_{i}{ext}"

    return new_filepath
