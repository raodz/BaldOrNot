import os
import cv2
import numpy as np

from src.constants import DEFAULT_IMG_SIZE
from src.model import BaldOrNotModel


def load_image_to_predict(img_dir_path: str, img_name: str) -> np.ndarray:
    """Load an image to predict from a given path."""
    img_path = os.path.join(img_dir_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_name}")
    return img


