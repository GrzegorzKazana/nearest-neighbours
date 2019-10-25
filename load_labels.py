import pandas as pd
import numpy as np

def get_labels_by_filename(data: pd.DataFrame, filename: str) -> np.ndarray:
    labels = data[data["filename"] == filename]
    labels.drop(columns=["filename"], inplace=True)
    return labels.to_numpy()
