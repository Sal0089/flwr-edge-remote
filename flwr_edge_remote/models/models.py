"""File collecting all model implementations."""


import torch
import torch.nn as nn
import torch.nn.functional as F

# Current version of the CNN
class CNN(nn.Module):
    """Balanced architecture for FL on the P2S dataset."""

    # Definition of the network layers
    def __init__(self, input_length=4096):
        super().__init__()  # Inherits from nn.Module
        self.supports_mask = True  # Indicates that this model does not use dataset["mask"]
        
        # Feature extraction with progressively smaller kernels
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.gn1 = nn.GroupNorm(8, 32)  # Normalization for improved robustness
        self.pool1 = nn.MaxPool1d(2)  # Reduces input length to 2048
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool1d(2)  # Reduces input length to 1024
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.gn3 = nn.GroupNorm(16, 128)
        self.pool3 = nn.MaxPool1d(2)  # Reduces input length to 512
        
        # Performs Global Average Pooling (computes mean across the temporal dimension for each filter)
        self.gap = nn.AdaptiveAvgPool1d(1)  # 128 output values
        
        # Classifier with regularization
        self.fc1 = nn.Linear(128, 64)
        # Randomly deactivates 50% of neurons to prevent overfitting
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    # Forward pass
    def forward(self, x, mask=None):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


# CNN implementing an attention layer that leverages an annotation mask
# to handle confounding regions during the forward pass
class CNNWithAttention(nn.Module):
    """
    CNN that uses the mask to down-weight features
    from confusing regions during the forward pass.
    """
    def __init__(self, input_length=4096):
        super().__init__()
        self.supports_mask = True
        
        # Feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool1d(2)  # -> 2048
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool1d(2)  # -> 1024
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.gn3 = nn.GroupNorm(16, 128)
        self.pool3 = nn.MaxPool1d(2)  # -> 512
        
        # Mask-aware attention layer
        self.mask_attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        # Feature extraction
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 128, 512)
        
        # MASK-AWARE ATTENTION
        if mask is not None:
            # Downsample mask to match feature dimensions
            mask_down = F.interpolate(mask, size=x.shape[2], mode='nearest')
            
            # Create attention weights
            # mask=1 (confusing region) -> low weight (0.3)
            # mask=0 (clean region) -> high weight (1.0)
            attention_weights = 1.0 - 0.7 * mask_down  # [0.3, 1.0]
            
            # Combine with learnable attention
            learned_attention = self.mask_attention(x)
            combined_attention = learned_attention * attention_weights
            
            # Apply attention
            x = x * combined_attention
        
        # Classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x