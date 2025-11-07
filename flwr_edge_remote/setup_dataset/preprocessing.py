"""Preprocessing utilities for AIML-TUDA/P2S dataset."""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import StandardScaler


def normalize_time_series(sample):
    """
    Normalize a single time series using z-score standardization:
    (x - mean) / std.
    If the standard deviation is zero, the original signal is returned unchanged.
    """
    sample = np.array(sample)
    std = np.std(sample)
    if std == 0:
        return sample
    return (sample - np.mean(sample)) / std


def normalization(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Apply normalization to the main features of the AIML-TUDA/P2S dataset.
    
    - dowel_deep_drawing_ow: sample-wise z-score standardization (time series)
    - mask: left unchanged (binary annotation)
    """
    print("Normalizing data...")

    # Sample-wise normalization for time series (applied to both partitions)
    for df in [train_df, test_df]:
        df["dowel_deep_drawing_ow"] = df["dowel_deep_drawing_ow"].apply(normalize_time_series)

    print("Normalization completed.")
    return train_df, test_df


def data_augmentation(dataset: pd.DataFrame, noise_level: float = 0.01):
    """
    Add Gaussian noise to the time series (to be used only on the training set).
    Other augmentations such as time warping or scaling can be added later.
    """
    dataset["dowel_deep_drawing_ow"] = dataset["dowel_deep_drawing_ow"].apply(
        lambda ts: np.array(ts) + np.random.normal(0, noise_level, len(ts))
    )
    return dataset


def preprocess(trainset, testset, apply_augmentation: bool = False, to_torch: bool = True):
    """
    Execute the preprocessing pipeline for the AIML-TUDA/P2S dataset.

    Steps:
    - Normalize the time series sample-wise
    - Optionally apply data augmentation on the training set if apply_augmentation=True
    - Convert (optionally) the dataset to PyTorch format (default: True)

    The function accepts either a HuggingFace Dataset or a Pandas DataFrame.
    It returns the dataset in the same format as the input.
    """

    # Detect input format
    is_hf_trainset = not isinstance(trainset, pd.DataFrame)
    is_hf_testset = not isinstance(testset, pd.DataFrame)

    # Convert to Pandas if needed
    if is_hf_trainset:
        trainset = trainset.to_pandas()
    if is_hf_testset:
        testset = testset.to_pandas()

    # Perform normalization consistently across train and test
    trainset, testset = normalization(trainset, testset)

    # Optional data augmentation (only on the training set)
    if apply_augmentation:
        trainset = data_augmentation(trainset)

    # Convert back to HuggingFace Dataset if required
    if is_hf_trainset:
        trainset = Dataset.from_pandas(trainset)
    if is_hf_testset:
        testset = Dataset.from_pandas(testset)

    # Convert to PyTorch format
    if to_torch:
        columns = ["dowel_deep_drawing_ow", "label", "speed", "mask"] # Cambia in speed_norm se normalizzi
        trainset.set_format(type="torch", columns=columns)
        testset.set_format(type="torch", columns=columns)

    return trainset, testset
