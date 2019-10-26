import os
import numpy as np


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


def serialize_labels_with_images(labels_with_images, labels_path, images_path):
    labels, imgs = zip(*labels_with_images)

    labels_np = np.array(labels)
    imgs_np = np.array(imgs)

    np.save(labels_path, labels_np)
    np.save(images_path, imgs_np)
