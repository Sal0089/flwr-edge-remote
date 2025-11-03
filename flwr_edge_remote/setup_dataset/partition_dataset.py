"""
partition_dataset.py
Downloads the AIML-TUDA/P2S dataset and partitions it in a non-IID manner.
"""

import os
import numpy as np
from datasets import load_dataset

# Configuration parameters
DATASET_NAME = "AIML-TUDA/P2S"
CONFIG_NAME = "Decoy"  # Possible values: [Normal, Decoy]
NUM_PARTITIONS = 3
ALPHA = 0.3  # Dirichlet concentration parameter
OUTPUT_DIR = "./flwr_edge_remote/dataset_partitions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the AIML-TUDA/P2S dataset
print("Loading dataset...")
dataset_path = "./flwr_edge_remote/dataset"

if os.path.exists(dataset_path):
    ds = load_dataset(dataset_path, split="train")
else:
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split="all")

print(ds)
print(f"Dataset loaded with {len(ds)} examples.")

# Partition the dataset using a custom Dirichlet-based partitioner
def dirichlet_split_by_speed_and_label(dataset, num_partitions, alpha=1.0):
    """
    Partitioning based on both label and speed.
    Goal: create clients with heterogeneous speed distributions.
    """
    import pandas as pd
    
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        'label': dataset['label'],
        'speed': dataset['speed'],
        'index': range(len(dataset))
    })
    
    # Create "speed_label" groups (e.g., 80_Normal, 80_Defect, ...)
    df['group'] = df['speed'].astype(str) + '_' + df['label'].astype(str)
    
    partitions = [[] for _ in range(num_partitions)]
    
    # Apply Dirichlet sampling to each (speed, label) group
    for group_name, group_df in df.groupby('group'):
        indices = group_df['index'].values
        np.random.shuffle(indices)
        
        proportions = np.random.dirichlet(alpha * np.ones(num_partitions))
        proportions = (proportions / proportions.sum()) * len(indices)
        split_points = np.cumsum(proportions).astype(int)[:-1]
        split_indices = np.split(indices, split_points)
        
        for i, part in enumerate(split_indices):
            partitions[i].extend(part)
    
    # Print statistics for verification
    for i, indices in enumerate(partitions):
        part_df = df[df['index'].isin(indices)]
        print(f"\nClient {i}:")
        print(f"  Total samples: {len(indices)}")
        print(f"  Speed distribution:\n{part_df['speed'].value_counts().sort_index()}")
        print(f"  Label distribution: Normal={sum(part_df['label']==0)}, Defect={sum(part_df['label']==1)}")
    
    return partitions


print("Performing non-IID partitioning (Dirichlet)...")
partitions = dirichlet_split_by_speed_and_label(ds, NUM_PARTITIONS, ALPHA)


# Split each partition into train and test sets
def split_train_test(dataset, test_ratio=0.2, seed=42):
    """Split a dataset into train and test subsets randomly."""
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    test_size = int(len(dataset) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return dataset.select(train_indices), dataset.select(test_indices)


# Save partitions to disk
for i, indices in enumerate(partitions):
    part_ds = ds.select(indices)
    part_dir = os.path.join(OUTPUT_DIR, f"partition_{i}")
    os.makedirs(part_dir, exist_ok=True)

    # Split: 80% train / 20% test
    train_ds, test_ds = split_train_test(part_ds, test_ratio=0.2)

    # Save both subsets
    train_ds.save_to_disk(os.path.join(part_dir, "train"))
    test_ds.save_to_disk(os.path.join(part_dir, "test"))

    print(f"Partition {i} saved with {len(train_ds)} train and {len(test_ds)} test examples → {part_dir}")

print("Dataset successfully partitioned and saved.")
