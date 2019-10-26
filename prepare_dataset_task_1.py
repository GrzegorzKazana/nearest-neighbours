import os
import pandas as pd
import numpy as np

from load_labels import get_labels_by_filename
from transform import load_image_to_numpy
from constants import OBJECT_LABELS
from helpers import get_paths, get_paths_and_names, filter_non_existent_paths

data_path = "./data"
labels_file = "labels.csv"
images_dir = "main_task_data"
labels_np_file = "labels_task1.npy"
images_np_file = "imgs_task1.npy"


def main():
    df = pd.read_csv(os.path.join(data_path, labels_file), usecols=["filename", *OBJECT_LABELS])
    img_paths = get_paths(os.path.join(data_path, images_dir), ".jpg")
    path_w_names = get_paths_and_names(img_paths)
    path_w_names = filter_non_existent_paths(path_w_names, df)

    labels_w_imgs = [(get_labels_by_filename(df, name), load_image_to_numpy(path)) for path, name in path_w_names]

    labels, imgs = zip(*labels_w_imgs)

    labels_np = np.array(labels)
    imgs_np = np.array(imgs)

    np.save(os.path.join(data_path, labels_np_file), labels_np)
    np.save(os.path.join(data_path, images_np_file), imgs_np)


if __name__ == "__main__":
    main()
