"""
task.py — Federated Learning client utilities

This module defines core functions used by each client in a federated learning setup.
It provides dataset loading, preprocessing, model training, and evaluation routines,
with optional support for mask-aware models and speed-robustness analysis.
"""


import torch
import numpy as np 
import random
from pathlib import Path
from torch import nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from flwr_edge_remote.setup_dataset.preprocessing import preprocess


# Utility functions for loading (and caching) dataloaders
_cached_loaders = {}
def load_client_dataset(client_id, batch_size=32):
    dataset_dir = Path("/home/salvo/desktop/venv/flwr_projects/flwr-edge-remote/flwr_edge_remote/dataset_partitions")
    train_path = dataset_dir / f"partition_{client_id}" / "train"
    test_path = dataset_dir / f"partition_{client_id}" / "test"

    print(f"[DEBUG] client_id={client_id} -> using partition_{client_id}")
    print(f"[DEBUG] train_path={train_path}")
    print(f"[DEBUG] test_path={test_path}")

    trainset = load_from_disk(str(train_path))
    testset = load_from_disk(str(test_path))
    trainset, testset = preprocess(trainset, testset, apply_augmentation=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def get_or_load_dataloaders(client_id, batch_size):
    """Load dataloaders from cache if available, otherwise create and store them."""
    if client_id not in _cached_loaders:
        _cached_loaders[client_id] = load_client_dataset(client_id, batch_size=batch_size)
    return _cached_loaders[client_id]


def test_speed_robustness_detailed(model, test_loader, device):
    """
    Measures the model's robustness across different operating speeds.
    
    Evaluates classification accuracy per speed level and computes
    summary statistics such as mean, standard deviation, and accuracy range.
    
    Returns:
        dict: Robustness metrics including:
            - 'per_speed': accuracy per speed value
            - 'mean': mean accuracy across speeds
            - 'std': standard deviation of accuracy
            - 'range': variation between max and min accuracy
    """
    from collections import defaultdict
    
    model.eval()
    results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["dowel_deep_drawing_ow"].to(device).unsqueeze(1).float()
            y = batch["label"].to(device).float().view(-1)
            speed = batch["speed"]
            preds = (model(x) > 0.5).float().squeeze()
            
            for i in range(len(y)):
                s = int(speed[i].item())
                results[s]['total'] += 1
                if preds[i] == y[i]:
                    results[s]['correct'] += 1
    
    # Compute per-speed accuracies
    accs = {speed: stats['correct'] / stats['total']
            for speed, stats in results.items() if stats['total'] > 0}
    
    if not accs:
        return {'per_speed': {}, 'mean': 0.0, 'std': 0.0, 'range': 0.0}
    
    acc_values = list(accs.values())
    return {
        'per_speed': accs,
        'mean': float(np.mean(acc_values)),
        'std': float(np.std(acc_values)),
        'range': float(np.max(acc_values) - np.min(acc_values))
    }


# Training and evaluation functions
def train(net, trainloader, epochs, learning_rate, device):
    """Train the model."""
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    net.train()

    running_loss = 0.0
    mask_dropout_rate = 0.3  # Probability to drop mask during training

    for _ in range(epochs):
        for batch in trainloader:
            x = batch["dowel_deep_drawing_ow"].to(device).float().unsqueeze(1)
            y = batch["label"].to(device).float().unsqueeze(1)

            # Pass mask to the model if supported
            mask = None
            if getattr(net, "supports_mask", False):
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device).float().unsqueeze(1)

                    # Apply dropout on mask elements (drop individual samples, not whole batch)
                    dropout_mask = (torch.rand_like(mask) > mask_dropout_rate).float()
                    mask = mask * dropout_mask

                output = net(x, mask=mask)
            else:
                output = net(x)

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate the model, optionally supporting mask-based evaluation."""
    net.to(device)
    net.eval()
    criterion = nn.BCELoss()

    total_loss = 0.0
    correct = 0

    correct_with_mask = 0
    correct_without_mask = 0
    
    with torch.no_grad():
        for batch in testloader:
            x = batch["dowel_deep_drawing_ow"].to(device).float().unsqueeze(1)
            y = batch["label"].to(device).float().unsqueeze(1)
            
            # Pass mask if the model supports it
            mask = None
            if hasattr(net, 'supports_mask') and net.supports_mask:
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device).float().unsqueeze(1)
                # Forward pass (with or without mask)
                outputs = net(x, mask=mask)
            else:
                outputs = net(x)
            
            # Compute global loss and accuracy
            total_loss += criterion(outputs, y).item()
            preds = (outputs > 0.5).float()
            correct += (preds == y).sum().item()
            
            # Compute masked and unmasked accuracy 
            if mask is not None:
                mask = mask.to(device).float()
                
                # With mask guidance (which is provided in the model with attention)
                preds_with = (net(x, mask) > 0.5).float().squeeze()
                correct_with_mask += (preds_with == y).sum().item()
                
                # Without mask (blind)
                preds_without = (net(x, mask=None) > 0.5).float().squeeze()
                correct_without_mask += (preds_without == y).sum().item()
            else:
                # No mask available
                preds = (net(x) > 0.5).float().squeeze()
                correct_with_mask += (preds == y).sum().item()
                correct_without_mask += (preds == y).sum().item()

    eval_loss = total_loss / len(testloader)
    eval_acc = correct / len(testloader.dataset)

    # Computes mask benefit
    masked_acc = correct_with_mask / len(testloader.dataset)
    unmasked_acc = correct_without_mask / len(testloader.dataset)
    mask_benefit = masked_acc - unmasked_acc

    # Compute robustness to speed variation
    speed_stats = test_speed_robustness_detailed(net, testloader, device)

    return eval_loss, eval_acc, mask_benefit, speed_stats
