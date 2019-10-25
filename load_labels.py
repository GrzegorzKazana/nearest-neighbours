import pandas as pd
import numpy as np

LABELS_DATA_PATH = "./data/labels.csv"

OBJECT_LABELS = [
    "Bathroom",
    "Bathroom cabinet",
    "Bathroom sink",
    "Bathtub",
    "Bed",
    "Bed frame",
    "Bed sheet",
    "Bedroom",
    "Cabinetry",
    "Ceiling",
    "Chair",
    "Chandelier",
    "Chest of drawers",
    "Coffee table",
    "Couch",
    "Countertop",
    "Cupboard",
    "Curtain",
    "Dining room",
    "Door",
    "Drawer",
    "Facade",
    "Fireplace",
    "Floor",
    "Furniture",
    "Grass",
    "Hardwood",
    "House",
    "Kitchen",
    "Kitchen & dining room table",
    "Kitchen stove",
    "Living room",
    "Mattress",
    "Nightstand",
    "Plumbing fixture",
    "Property",
    "Real estate",
    "Refrigerator",
    "Roof",
    "Room",
    "Rural area",
    "Shower",
    "Sink",
    "Sky",
    "Table",
    "Tablecloth",
    "Tap",
    "Tile",
    "Toilet",
    "Tree",
    "Urban area",
    "Wall",
    "Window",
    "filename",
]


def get_labels_by_filename(data: pd.DataFrame, filename: str) -> np.ndarray:
    labels = data[data["filename"] == filename]
    labels.drop(columns=["filename"], inplace=True)
    return labels.to_numpy()


def load_objects_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_DATA_PATH, usecols=OBJECT_LABELS)
    return df
