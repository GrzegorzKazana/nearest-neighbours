import pandas as pd
import numpy as np
from constants import TASK2_CLASS_LABELS


def get_labels_by_filename(data: pd.DataFrame, filename: str) -> np.ndarray:
    print(filename)
    labels = data[data["filename"] == filename].copy()
    labels.drop(columns=["filename"], inplace=True)
    return labels.to_numpy()


def get_room_labels_by_filename(data: pd.DataFrame, filename: str) -> np.ndarray:
    print(filename)
    label = data[data["filename"] == filename].iloc[0, 1]
    ohe = np.zeros(len(TASK2_CLASS_LABELS))
    ohe[TASK2_CLASS_LABELS.index(label)] = 1
    return ohe
