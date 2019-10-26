import os


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
