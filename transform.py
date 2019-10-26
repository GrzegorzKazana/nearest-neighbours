import typing
import pathlib

import cv2
import numpy as np

DEFAULT_SIZE_LENGTH = 224

# def mirror_fill(image: np.ndarray, desired_shape: typing.Tuple[int, int]) -> np.ndarray:
#     ...


def mirror_fill(image: np.ndarray, side_length: int) -> np.ndarray:
    height, width, channels_count = image.shape
    max_dim = max(height, width)
    transformed_image = np.zeros((max_dim, max_dim, channels_count))
    if height > width:
        transformed_image[:, :width, :] = image
        transformed_image[:, width:, :] = image[:, width : 2 * width - max_dim - 1 : -1, :]
    else:
        transformed_image[:height, :, :] = image
        transformed_image[height:, :, :] = image[height : 2 * height - max_dim - 1 : -1, :, :]
    return cv2.resize(transformed_image * 1 / 256, (side_length, side_length))


def stretch(image: np.ndarray, desired_shape: typing.Tuple[int, int]) -> np.ndarray:
    # print(image.shape, desired_shape)


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
    image = load_image_to_numpy(pathlib.Path("img3.jpg"), resizing_strategy=mirror_fill, side_length=400)
    show_image(image)
