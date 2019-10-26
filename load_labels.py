import pandas as pd
import numpy as np


def get_labels_by_filename(data: pd.DataFrame, filename: str) -> np.ndarray:
    print(filename)
    labels = data[data["filename"] == filename].copy()
    labels.drop(columns=["filename"], inplace=True)
    return labels.to_numpy()
