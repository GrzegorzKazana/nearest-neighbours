import os
import numpy as np

from transform import load_image_to_numpy
from helpers import get_paths, get_paths_and_names

data_path = "./data"
test_data_dir = "test_dataset"
test_images_np = "test_images.npy"
test_images_names = "test_images_names.txt"


test_data_path = os.path.join(data_path, test_data_dir)
test_images_np_path = os.path.join(data_path, test_images_np)
test_images_names_path = os.path.join(data_path, test_images_names)

img_paths = get_paths(os.path.join(data_path, test_data_dir), ".jpg")
path_w_names = get_paths_and_names(img_paths)

IMAGE_SIDE_LENGTH = 128


def main():
    names, images = zip(*[
        (name, load_image_to_numpy(path, IMAGE_SIDE_LENGTH))
        for path, name in path_w_names
    ])

    with open(test_images_names_path, "w") as file:
        file.write("\n".join(names))

    images_np = np.array(images)
    np.save(test_images_np_path, images_np)


if __name__ == '__main__':
    main()

