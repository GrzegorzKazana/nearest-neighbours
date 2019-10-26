import os
import sys
import pandas as pd
import numpy as np
from load_labels import get_labels_by_filename
from transform import load_image_to_numpy
from constants import OBJECT_LABELS

data_path = "./data"
labels_file = "labels.csv"
images_dir = "main_task_data"


def get_paths(dir_path, extension):
    path_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(extension):
                path_list.append(os.path.join(root, file))

    return path_list


def get_paths_and_names(paths):
    return [(p, os.path.basename(p)) for p in paths]


def filter_non_existent_paths(paths_w_names, df):
    return [(p, n) for p, n in paths_w_names if n in df["filename"].to_numpy()]


def main():
    df = pd.read_csv(os.path.join(data_path, labels_file), usecols=["filename", *OBJECT_LABELS])
    img_paths = get_paths(os.path.join(data_path, images_dir), ".jpg")
    path_w_names = get_paths_and_names(img_paths)
    path_w_names = filter_non_existent_paths(path_w_names, df)

    print(path_w_names)

    labels_w_imgs = [(get_labels_by_filename(df, name), load_image_to_numpy(path)) for path, name in path_w_names]

    labels, imgs = zip(*labels_w_imgs)

    labels_np = np.array(labels)
    imgs_np = np.array(imgs)

    np.save(os.path.join(data_path, "labels_task1.npy"), labels_np)
    np.save(os.path.join(data_path, "imgs_task1.npy"), imgs_np)


if __name__ == "__main__":
    main()
