import os
import cv2
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from src.config import BoldOrNotConfig


def check_sample_images(directory: str) -> Tuple[List[str], int]:
    """
    Checks the images in the specified directory to identify empty or corrupted files.

    Args:
        directory (str): The path to the directory containing the images.

    Returns:
        Tuple[List[str], int, int]:
            - A list of filenames that are either empty or corrupted.
            - The count of empty or corrupted images.
            - The count of correctly loaded images.

    This function iterates through all the files in the given directory, attempting to load each image using OpenCV's `
    cv2.imread` function.
    If an image cannot be loaded (i.e., it is empty or corrupted), it is added to the `empty_or_corrupted` list.
    The function finally returns this list along with the count of corrupted/empty images and the count of successfully
    loaded images.
    """
    empty_or_corrupted: List[str] = []
    num_correct: int = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            empty_or_corrupted.append(filename)
        else:
            num_correct += 1

    return empty_or_corrupted, num_correct


def prepare_merged_dataframe(
    subsets_path: str, labels_path: str
) -> pd.DataFrame:
    """
    Prepares a combined DataFrame by merging two CSV files on a common column.

    This function reads two CSV files into DataFrames: one containing subsets of data
    and the other containing labels. It then merges these DataFrames on the "image_id"
    column using an inner join, which means that only the rows with matching "image_id"
    values in both DataFrames will be retained in the final result.

    Parameters:
    subsets_path (str): The file path to the CSV containing the subsets data.
    labels_path (str): The file path to the CSV containing the labels data.

    Returns:
    pd.DataFrame: A merged DataFrame containing data from both input CSVs,
                  joined on the "image_id" column.
    """
    subsets = pd.read_csv(subsets_path)
    labels = pd.read_csv(labels_path)
    return pd.merge(subsets, labels, how="inner", on="image_id")


def create_data_subsets(df: pd.DataFrame) -> None:
    """
    Splits the provided DataFrame into training, validation, and test datasets, and saves them as CSV files.

    Args:
        df (pd.DataFrame): A DataFrame containing the dataset, which includes image IDs, partition labels,
        and other relevant data.

    Returns:
        None: This function does not return any value but saves three CSV files: `train.csv`, `validation.csv`,
        and `test.csv`.

    This function processes the input DataFrame by first filtering the data into training and test sets
    based on the `partition` column.
    The training set (where `partition` is 0) is further split into training and validation subsets,
    with 9% of the data allocated
    to the validation set. The resulting subsets are then saved as separate CSV files:
    - `train.csv` for the training data
    - `validation.csv` for the validation data
    - `test.csv` for the test data

    The function also prints the number of samples in each of these subsets.
    """

    train_df = df[df["partition"] == 0]
    test_df = df[df["partition"] == 1].drop(columns=["partition"])
    train_df, val_df = train_test_split(
        train_df.drop(columns=["partition"]), test_size=0.09, random_state=42
    )

    print("Number of samples in the training set:", len(train_df))
    print("Number of samples in the validation set:", len(val_df))
    print("Number of samples in the test set:", len(test_df))

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)


class BaldDataset(tf.keras.utils.Sequence):
    """
    Generates random image data for Keras models using TensorFlow tensors.

    This dataset class simulates a dataset for training purposes, generating random images
    with dimensions 224x224x3 and corresponding random binary labels.

    Attributes:
        num_samples (int): The total number of samples in the dataset.
        batch_size (int): The number of samples per batch.
        vector_dim (int): The dimensionality of the random vectors (unused in this context).
    """
    def __init__(self, num_samples: int = 100, batch_size: int = 10, vector_dim: int = 100):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return int(tf.math.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing a batch of images (X) and labels (y).
        """
        indexes = tf.range(index * self.batch_size, (index + 1) * self.batch_size)
        X = tf.random.normal(shape=(len(indexes), 224, 224, 3))
        y = tf.random.uniform(shape=(len(indexes), 1), minval=0, maxval=2, dtype=tf.int32)
        return X, y



