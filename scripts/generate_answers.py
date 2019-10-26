import csv
import os
import logging
import random
import time
import argparse
from typing import Tuple

__author__ = "ING_DS_TECH"
__version__ = "201909"

FORMAT = "%(asctime)-15s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

labels_task_1 = [
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
]

labels_task2 = ["apartment", "bathroom", "bedroom", "dinning_room", "house", "kitchen", "living_room"]

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]

output = []


def generate_answers_filename(path: str) -> str:
    return f"{path}/answer-{time.time()}.txt"


def task_1(partial_output: dict, file_path: str) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))

    for label in labels_task_1:
        partial_output[label] = 0
    #
    #
    # 	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(file_path: str) -> str:
    logger.debug("Performing task 2 for file {0}".format(file_path))
    #
    #
    # 	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return labels_task2[random.randrange(len(labels_task2))]


def task_3(file_path: str) -> Tuple[str, str]:
    logger.debug("Performing task 3 for file {0}".format(file_path))
    #
    #
    # 	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return labels_task3_1[random.randrange(len(labels_task3_1))], labels_task3_2[random.randrange(len(labels_task3_2))]


def main(input_path: str, output_path: str):
    logger.debug("Sample answers file generator")
    for dirpath, dnames, fnames in os.walk(input_path):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                output_per_file = {
                    "filename": f,
                    "task2_class": task_2(file_path),
                    "tech_cond": task_3(file_path)[0],
                    "standard": task_3(file_path)[1],
                }
                output_per_file = task_1(output_per_file, file_path)

                output.append(output_per_file)

    with open(output_path, "w+", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["filename", "standard", "task2_class", "tech_cond"] + labels_task_1
        )
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Folder with images")
    parser.add_argument("output", type=str, help="Answer file directory")
    args = parser.parse_args()
    input_path = args.input
    output_path = generate_answers_filename(args.output)
    main(input_path, output_path)
