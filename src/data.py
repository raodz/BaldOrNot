import os
from collections import Counter
from typing import List, Tuple

import cv2
import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from src.config_class import BaldOrNotConfig
from src.constants import (
    DEFAULT_IMG_SIZE,
    N_CHANNELS_GRAYSCALE,
    N_CHANNELS_RGB,
)
from src.exceptions import BaldOrNotDataError


class BaldDataset(keras.utils.Sequence):
    """
    Generates data for Keras models.

    This class is responsible for creating batches of data to be fed into
    a Keras model.
    It loads images from a directory, preprocesses them, and returns them
    along with their corresponding labels.

    Attributes:
    ----------
    df : pd.DataFrame
        The DataFrame containing the image IDs and labels.
    batch_size : int
        The number of samples per batch.
    dim : Tuple[int, int]
        The dimensions to which all images will be resized (height, width).
    n_channels : int, optional
    Number of channels in the images. Must be either 1 for grayscale or 3 for
    RGB images (default is 3).
    n_classes : int
        The number of classes (used for one-hot encoding of labels).
    shuffle : bool
        Whether to shuffle the order of samples at the end of each epoch.
    indexes : np.ndarray
        Array of indices used to keep track of the current batch.
    labels : pd.Series
        Series containing labels corresponding to each image ID.
    list_IDs : pd.Series
        Series containing the list of image IDs.

    Methods:
    -------
    __len__():
        Returns the number of batches per epoch.
    __getitem__(index: int) -> Tuple[np.ndarray, np.ndarray]:
        Generates one batch of data.
    on_epoch_end():
        Updates indexes after each epoch.
    __data_generation(list_IDs_temp: List[str]) ->
    Tuple[np.ndarray, np.ndarray]:
        Generates data containing batch_size samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        dim: Tuple[int, int] = DEFAULT_IMG_SIZE,
        n_channels: int = N_CHANNELS_RGB,
        shuffle: bool = True,
    ):
        """
        Initialization method for BaldDataset.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing image IDs and labels.
        batch_size : int, optional
            The number of samples per batch (default is 32).
        dim : Tuple[int, int], optional
            Dimensions to which images will be resized (default is (218, 178)).
        n_channels : int, optional
            Number of channels in the images (default is 3 for RGB).
        n_classes : int, optional
            Number of classes for classification (default is 2).
        shuffle : bool, optional
            Whether to shuffle the data at the beginning of each epoch
            (default is True).
        """
        super().__init__()
        if n_channels not in [N_CHANNELS_GRAYSCALE, N_CHANNELS_RGB]:
            raise ValueError(
                "n_channels must be either 1 (grayscale) or 3 (RGB)."
            )

        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.list_IDs = df["image_id"]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.config = BaldOrNotConfig()
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
        -------
        int
            Number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Parameters:
        ----------
        index : int
            Index of the batch.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            A batch of images and their corresponding labels.
        """
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_preprocessing(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.

        This method is called at the end of each epoch and is responsible for
        shuffling the data if the shuffle attribute is set to True.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_preprocessing(
        self, list_IDs_temp: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples.

        Parameters:
        ----------
        list_IDs_temp : List[str]
            List of image IDs to generate data for.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            A batch of images and their corresponding labels.
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        images_dir = self.config.paths.images_dir

        for i, ID in enumerate(list_IDs_temp):
            reconverted_ID = f"{int(ID):06d}.jpg"
            image_path = os.path.join(images_dir, reconverted_ID)  # noqa: E231
            image = cv2.imread(image_path)

            if image is None:
                raise BaldOrNotDataError(f"Failed to load image: {image_path}")

            image = cv2.resize(image, self.dim[::-1])
            # avoiding ValueError by adjusting self.dim order

            # If grayscale, convert it to RGB
            if (
                image.shape[-1] == N_CHANNELS_GRAYSCALE
                and self.n_channels == N_CHANNELS_RGB  # noqa: W503
            ):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            X[i] = image / 255.0  # Normalize to range [0, 1]
            y[i] = self.df.loc[self.df["image_id"] == ID, "labels"].values[0]

        return X, y

    @staticmethod
    def __get_wrong_files_list(directory: str) -> List[str]:
        """
        Scans a directory for empty or corrupted image files.

        Parameters:
        ----------
        directory : str
            The path to the directory containing the image files.

        Returns:
        -------
        List[str]
            A list of filenames that are empty or corrupted.
        """
        empty_or_corrupted: List[str] = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            img = cv2.imread(file_path)
            if img is None or img.size == 0:
                empty_or_corrupted.append(filename)

        return empty_or_corrupted

    @staticmethod
    def get_cleaned_df(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing rows corresponding to empty or
        corrupted images.

        Parameters:
        ----------
        df : pd.DataFrame
            The original DataFrame containing image IDs.
        images_dir : str
            The directory containing the images.

        Returns:
        -------
        pd.DataFrame
            A cleaned DataFrame with rows for empty or corrupted images
            removed.
        """
        empty_or_corrupted = BaldDataset.__get_wrong_files_list(images_dir)
        cleaned_df = df[~df["image_id"].isin(empty_or_corrupted)]

        return cleaned_df

    @staticmethod
    def prepare_merged_dataframe(
        subsets_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepares a combined DataFrame by merging two CSV files on a common
        column.

        This function reads two CSV files into DataFrames: one containing
        subsets of data and the other containing labels. It then merges these
        DataFrames on the "image_id"  column using an inner join, which means
        that only the rows with matching "image_id" values in both DataFrames
        will be retained in the final result.

        Parameters:
        ----------
        subsets_path : str
            The file path to the CSV containing the subsets data.
        labels_path : str
            The file path to the CSV containing the labels data.

        Returns:
        -------
        pd.DataFrame
            A merged DataFrame containing data from both input CSVs,
            joined on the "image_id" column.
        """
        df_merged = pd.merge(subsets_df, labels_df, how="inner", on="image_id")

        return df_merged[["image_id", "partition", "Bald"]].rename(
            columns={"Bald": "labels"}
        )

    @staticmethod
    def create_subset_dfs(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into training, validation, and test sets.

        This function filters the DataFrame based on the partition column and
        then  splits the training data further into training and validation
        sets.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to split.

        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Three DataFrames: train_df, val_df, and test_df.
        """
        train_encoding = 0
        test_encoding = 1
        train_df = df[df["partition"] == train_encoding]
        test_df = df[df["partition"] == test_encoding].drop(
            columns=["partition"]
        )
        train_df, val_df = train_test_split(
            train_df.drop(columns=["partition"]),
            test_size=0.09,
            random_state=42,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df

    @staticmethod
    def convert_image_id_column_to_float(
        df: pd.DataFrame, image_id_col: str = "image_id"
    ) -> pd.DataFrame:
        df[image_id_col] = (
            df[image_id_col].str.replace(".jpg", "", regex=False).astype(float)
        )
        return df

    @staticmethod
    def balance_classes(
        df: pd.DataFrame,
        X_cols: list,
        y_col: str,
        minor_class_multiplier: int = 10,
    ) -> pd.DataFrame:
        """
        Balances the classes in a DataFrame using a combination of oversampling
        and undersampling.

        This function oversamples the minority class using SMOTE by a factor
        specified by `minority_multiplier` and then undersamples the majority
        class using RandomUnderSampler to achieve a more balanced dataset.
        The feature columns specified in `X_cols` are retained along with the
        target column `y_col`.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data that needs to be balanced.
        X_cols : list
            A list of column names that represent the features (X).
        y_col : str
            The name of the column that contains the class labels (y).
        minority_multiplier : int
            The factor by which the minority class should be oversampled
            (default is 10).

        Returns:
        --------
        pd.DataFrame
            A balanced DataFrame with class distributions adjusted based on
            the multiplier.
        """

        # Split into features (X) and labels (y)
        X = df[X_cols]
        y = df[y_col]

        # logger.info(f"Number of samples before balancing: {Counter(y)}")

        # Get the number of samples in the minority class
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]

        # Calculate the desired size for the minority class after oversampling
        target_minority_size = minority_count * minor_class_multiplier

        # Set the oversampling strategy
        sampling_strategy = {minority_class: target_minority_size}

        # Oversampling the minority class using SMOTE
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)

        # Undersampling the majority class
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

        # logger.info(
        #     f"Number of samples after balancing: {Counter(y_resampled)}")

        # Recreate the DataFrame after resampling
        balanced_df = pd.DataFrame(X_resampled, columns=X_cols)
        balanced_df[y_col] = y_resampled

        return balanced_df
