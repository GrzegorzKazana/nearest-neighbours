import typing
import pathlib
import itertools

import cv2
import numpy as np

DEFAULT_SIZE_LENGTH = 224


def oscillate(image: np.ndarray, axis: str = "rows"):
    height, width, _ = image.shape

    if axis == "rows":
        oscillator = itertools.cycle(itertools.chain(range(0, height), range(height - 1, 0)))
        for row in oscillator:
            yield image[height - row - 1, :, :]
    elif axis == "columns":
        oscillator = itertools.cycle(itertools.chain(range(0, width), range(width - 1, 0)))
        for column in oscillator:
            yield image[:, width - column - 1, :]
    else:
        raise ValueError("Illegal axis")


def mirror_fill(image: np.ndarray, side_length: int) -> np.ndarray:
    height, width, channels_count = image.shape
    max_dim = max(height, width)
    transformed_image = np.zeros((max_dim, max_dim, channels_count))

    if height > width:
        transformed_image[:, :width, :] = image
        for i, entry in enumerate(itertools.islice(oscillate(image, axis="columns"), max_dim - width)):
            transformed_image[:, width + i, :] = entry
    else:
        transformed_image[:height, :, :] = image
        for i, entry in enumerate(itertools.islice(oscillate(image, axis="rows"), max_dim - height)):
            transformed_image[height + i, :, :] = entry

    return cv2.resize(transformed_image * (1.0 / 256), (side_length, side_length))


def stretch(image: np.ndarray, side_length: int) -> np.ndarray:
    return cv2.resize(image, (side_length, side_length))


def load_image_to_numpy(
    image_path: pathlib.Path,
    side_length: int = DEFAULT_SIZE_LENGTH,
    resizing_strategy: typing.Callable[[np.ndarray, int], np.ndarray] = stretch,
) -> np.ndarray:
    image: np.ndarray = cv2.imread(str(image_path))
    return resizing_strategy(image, side_length)


def show_image(image):
    cv2.imshow("display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = load_image_to_numpy(pathlib.Path("img4.jpg"), resizing_strategy=mirror_fill, side_length=400)
    show_image(image)
