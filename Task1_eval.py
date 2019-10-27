import pickle
import talos
import numpy as np
import os
from tensorflow.keras.models import load_model
import pandas as pd
from constants import ALL_COLS
from joblib import load
from constants import TASK2_CLASS_LABELS, ROOM_LABELS, ALL_COLS,OBJECT_LABELS
import time
from datetime import datetime


ALL_COLS_W_RES = ["filename", "standard", "task2_class", "tech_cond", *ALL_COLS]


images_file_path = "test_imgs.npy"
# labels_file_path = 'labels_task1_validation.npy'

def get_timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def main():

    images = np.load(images_file_path).astype(np.float32)

    model = load_model("best_model.h5")

    res = model.predict(images)

    res = res.round()

    data = {label: res[:, i].astype(int) for i, label in enumerate(OBJECT_LABELS)}

    df = pd.DataFrame(data)
    # df.drop(["Bathroom", "Bedroom", "Kitchen", "Living room", "Dining room"], axis=1, inplace=True)

    clf = load("clf_single.joblib")
    task2_pred = clf.predict(df)

    with open("test_files.txt") as fp:
        filenames = [f.strip() for f in fp.readlines()]

    label_pairs = {
        "bathroom": "Bathroom",
        "dining_room": "Dining room",
        "house": "House",
        "kitchen": "Kitchen",
        "living_room": "Living room",
        "bedroom": "Bedroom",
    }

    room_labels = {
        room_lab: [
            (
                1
                if TASK2_CLASS_LABELS[np.argmax(ohe)] in label_pairs
                and (label_pairs[TASK2_CLASS_LABELS[np.argmax(ohe)]] == room_lab)
                else 0
            )
            for ohe in task2_pred
        ]
        for room_lab in ROOM_LABELS
    }

    data_csv = {
        "filename": filenames,
        "standard": np.random.randint(3, 5, size=len(filenames)),
        "task2_class": [TASK2_CLASS_LABELS[np.argmax(ohe)] for ohe in task2_pred],
        "tech_cond": np.random.randint(3, 5, size=len(filenames)),
        **room_labels,
        **df,
    }

    res_df = pd.DataFrame(data_csv)[ALL_COLS_W_RES]
    filename = f"result_{get_timestamp()}.csv"
    res_df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
