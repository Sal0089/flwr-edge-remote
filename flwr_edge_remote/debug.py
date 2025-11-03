# verify_mask_distribution.py

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from pathlib import Path

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
    - speed: global standardization (fit on train, transform on test)
    - mask: left unchanged (binary annotation)
    """
    print("Normalizing data...")

    # Sample-wise normalization for time series (applied to both partitions)
    for df in [train_df, test_df]:
        df["dowel_deep_drawing_ow"] = df["dowel_deep_drawing_ow"].apply(normalize_time_series)

    # Global standardization of the speed feature (fit only on train)
    """
    scaler = StandardScaler()
    train_df["speed_norm"] = scaler.fit_transform(train_df[["speed"]])
    test_df["speed_norm"] = scaler.transform(test_df[["speed"]])
    """
    print("Normalization completed.")
    return train_df, test_df


def data_augmentation(dataset: pd.DataFrame, noise_level: float = 0.01):
    """
    Add Gaussian noise to the time series (to be used only on the training set).
    Other augmentations such as jittering, time warping, or scaling can be added later.
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
    - Normalize the speed feature globally (fit on train, transform on test)
    - Optionally apply data augmentation on the training set if apply_augmentation=True
    - Optionally convert the dataset to PyTorch format (default: True)

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

print("="*70)
print("MASK VERIFICATION")
print("="*70)

# 1. Verifica dataset originale
print("\n1. Original Dataset from HuggingFace:")
ds_original = load_dataset("AIML-TUDA/P2S", "Decoy", split="train")
print(f"   Total samples: {len(ds_original)}")

mask_sums_original = [sum(ds_original[i]['mask']) for i in range(len(ds_original))]
unique_sums_original = set(mask_sums_original)

print(f"   Unique mask sums: {unique_sums_original}")
print(f"   Samples with mask.sum()=0: {mask_sums_original.count(0)}")
print(f"   Samples with mask.sum()>0: {len(ds_original) - mask_sums_original.count(0)}")

# 2. Verifica dataset partizionato (Client 0)
print("\n2. Partitioned Dataset (Client 0 train):")
dataset_dir = Path("./flwr_projects/flwr-edge-remote/flwr_edge_remote/dataset_partitions")
train_path = dataset_dir / "partition_0" / "train"
print(train_path)

if train_path.exists():
    ds_partition = load_from_disk(str(train_path))
    print(f"   Total samples: {len(ds_partition)}")
    
    # Controlla se mask esiste
    sample_keys = ds_partition[0].keys()
    print(f"   Available keys: {sample_keys}")
    
    if 'mask' in sample_keys:
        mask_sums_partition = []
        for i in range(len(ds_partition)):
            mask = ds_partition[i]['mask']
            # Controlla tipo
            print(f"   Sample {i}: mask type = {type(mask)}, shape = {np.array(mask).shape if hasattr(mask, '__len__') else 'scalar'}")
            if i >= 5:
                break  # Solo primi 5 per debug
            
            mask_sum = sum(mask) if hasattr(mask, '__len__') else 0
            mask_sums_partition.append(mask_sum)
        
        print(f"\n   First 5 mask sums: {mask_sums_partition[:5]}")
    else:
        print("   ❌ 'mask' key NOT FOUND in partitioned dataset!")
else:
    print("   ❌ Partition path does not exist!")

# 3. Verifica preprocessing
print("\n3. After Preprocessing:")
if train_path.exists():
    
    trainset = load_from_disk(str(train_path))
    testset = load_from_disk(str(dataset_dir / "partition_0" / "test"))
    
    print(f"   Before preprocess - trainset keys: {trainset[0].keys()}")
    
    trainset, testset = preprocess(trainset, testset, apply_augmentation=False, to_torch=True)
    
    print(f"   After preprocess - format: {trainset.format}")
    
    # Prova ad accedere alla mask
    try:
        sample = trainset[0]
        if 'mask' in sample:
            mask = sample['mask']
            print(f"   Mask accessible: YES")
            print(f"   Mask type: {type(mask)}")
            print(f"   Mask sum: {mask.sum().item() if hasattr(mask, 'sum') else sum(mask)}")
        else:
            print(f"   ❌ Mask NOT in processed sample!")
            print(f"   Available keys: {sample.keys()}")
    except Exception as e:
        print(f"   ❌ Error accessing mask: {e}")

print("\n" + "="*70)