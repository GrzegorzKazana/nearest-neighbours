import typing
import pathlib

import cv2
import numpy as np


def mirror_fill(image: np.ndarray, desired_shape: typing.Tuple[int, int]) -> np.ndarray:
    ...


def stretch(image: np.ndarray, desired_shape: typing.Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, desired_shape)


def load_image_to_numpy(image_path: pathlib.Path,
                        desired_shape: typing.Tuple[int, int] = (224, 224),
                        resizing_strategy: typing.Callable[
                            [np.ndarray, typing.Tuple[int, int]], np.ndarray] = stretch) -> np.ndarray:
    image: np.ndarray = cv2.imread(str(image_path))
    return resizing_strategy(image, desired_shape)


def show_image(image):
    cv2.imshow("display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = load_image_to_numpy(pathlib.Path("img.jpg"), resizing_strategy=stretch)
    show_image(image)
