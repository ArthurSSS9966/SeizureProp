import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import logging
from s4d import S4D
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import math
import torch
import torch.nn as nn
import optuna
import os
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base class for all models with common functionality."""

    def __init__(self, input_dim: int, output_dim: int, device: str = 'cuda:0'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Validate device
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        self.criteria = nn.CrossEntropyLoss()

    def random_init(self):
        """Initialize weights using appropriate initialization methods."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'conv' in name:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def configure_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-5):
        """Configure optimizer with parameters."""
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


class LSTM(BaseModel):
    def __init__(self, input_dim: int, output_dim: int,
                 lr = 0.001, weight_decay=1e-5,
                 hidden_dim: int = 50, dropout: float = 0.2,
                 num_layers: int = 1, device: str = 'cuda:0'):
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 8)
        self.dropout2 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(8 * 23, output_dim)  # Note: 23 should be calculated based on input

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shape
        if x.shape[-1] != self.input_dim:
            x = x.transpose(1, 2)

        x = x.to(self.device)

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        lstm_out = F.relu(lstm_out)
        lstm_out = self.dropout1(lstm_out)

        # Fully connected layers
        output = F.relu(self.fc1(lstm_out))
        output = self.dropout2(output)
        output = self.flatten(output)
        output = F.softmax(self.fc2(output), dim=1)

        return output


class CNN1D(BaseModel):
    def __init__(self, input_dim: int, output_dim: int,
                 lr=0.001, weight_decay=1e-5,
                 hidden_dim: int = 128, dropout: float = 0.2,
                 kernel_size: int = 256, device: str = 'cuda:0'):
        super().__init__(input_dim, output_dim, device)

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim if i == 0 else hidden_dim // (2 ** (i - 1)),
                    out_channels=hidden_dim // (2 ** i),
                    kernel_size=kernel_size // (2 ** i),
                    padding='same'
                ),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ) for i in range(3)  # 3 conv layers
        ])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((hidden_dim // 4) * (kernel_size // 8), hidden_dim // 8)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 8, output_dim)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1).to('cuda:0')

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))

        # Fully connected layers
        x = self.global_pool(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x


class Wavenet(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, hidden_dim=128, weight_decay=1e-5,
                 dropout=0.2, kernel_size=32, dilation=1):
        super(Wavenet, self).__init__()

        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size // 2 - 1) * dilation
        self.padding3 = (kernel_size // 4 - 1) * dilation

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=0, dilation=dilation).to(
            'cuda:0')
        self.maxpool1 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout1 = nn.Dropout(dropout).to('cuda:0')

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=kernel_size // 2, padding=0,
                               dilation=dilation).to('cuda:0')
        self.maxpool2 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout2 = nn.Dropout(dropout).to('cuda:0')

        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=kernel_size // 4, padding=0,
                               dilation=dilation).to('cuda:0')
        self.maxpool3 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout3 = nn.Dropout(dropout).to('cuda:0')

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1).to('cuda:0')

        # FC layers with fixed size due to global pooling
        self.fc1 = nn.Linear(hidden_dim // 4, hidden_dim // 8).to('cuda:0')
        self.dropout4 = nn.Dropout(dropout).to('cuda:0')
        self.fc2 = nn.Linear(hidden_dim // 8, output_dim).to('cuda:0')

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.to('cuda:0')

        # Convolutional layers
        x = F.pad(x, (self.padding1, 0))
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = F.pad(x, (self.padding2, 0))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = F.pad(x, (self.padding3, 0))
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def random_init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def predict(self, x):
        self.eval()
        x = x.to('cuda:0')
        output = self(x)
        return output


class S4Model(nn.Module):
    '''
    S4 model for seizure detection
    adapted from https://github.com/state-spaces/s4/blob/main/example.py#L153
    '''
    def __init__(
            self,
            d_input,  # Number of sEEG channels
            d_output=1,  # Binary classification per channel/timepoint
            d_model=64,  # Hidden dimension
            n_layers=1,  # Number of S4 layers
            dropout=0.2,
            prenorm=False,  # Apply normalization before or after S4 layer
            lr=0.001,  # Learning rate for S4D parameters
            weight_decay=1e-5
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder: project sEEG channels to model dimension
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers with residual connections
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(p=dropout))

        # Linear decoder for seizure detection
        self.decoder = nn.Linear(d_model, d_output)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.encoder(x)

        # Convert to (B, d_model, L) for S4Block
        x = x.transpose(1, 2)

        for i, (layer, norm, dropout) in enumerate(zip(self.s4_layers, self.norms, self.dropouts)):
            residual = x

            if self.prenorm:
                # Pre-normalization - need to handle transposed format
                x = norm(x.transpose(1, 2)).transpose(1, 2)

            x, _ = layer(x)

            x = dropout(x)

            x = x + residual

            if not self.prenorm:
                # Post-normalization - need to handle transposed format
                x = norm(x.transpose(1, 2)).transpose(1, 2)

        # Convert back to (B, L, d_model) for pooling
        x = x.transpose(1, 2)

        x = x.mean(dim=1)

        x = self.decoder(x)

        x = F.softmax(x, dim=1)

        return x

    def random_init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def predict(self, x):
        self.eval()
        x = x.to('cuda:0')
        output = self(x)
        return output


class ResNet(nn.Module):
    def __init__(
            self,
            input_dim=12,  # Number of input features
            base_filters=32,
            n_blocks=3,
            kernel_size=16,
            n_classes=2,
            dropout=0.2,
            lr=0.001,
            weight_decay=1e-5,
            norm_type='batch'
    ):
        super().__init__()

        self.norm_layer = nn.BatchNorm1d if norm_type == 'batch' else nn.LayerNorm

        # Initial feature embedding layer to process all features together
        self.feature_embedding = nn.Linear(input_dim, base_filters)

        # Initial causal conv - works on the time dimension after feature embedding
        self.initial = nn.Sequential(
            nn.Conv1d(base_filters, base_filters, kernel_size),
            self.norm_layer(base_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stack of dilated convolution blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 8)
            self.blocks.append(nn.ModuleDict({
                'dilated_conv': nn.Conv1d(
                    base_filters,
                    base_filters * 2,
                    kernel_size,
                    dilation=dilation,
                ),
                'norm': self.norm_layer(base_filters),
                'res_conv': nn.Conv1d(base_filters, base_filters, 1),
                'dropout': nn.Dropout(dropout)
            }))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Input: (batch_size, features, time_steps)
        Output: (batch_size, n_classes)
        """
        batch_size, n_features, time_steps = x.shape

        # Reshape and transpose to apply feature embedding
        # From [batch, features, time] to [batch * time, features]
        x = x.permute(0, 2, 1).reshape(batch_size * time_steps, n_features)

        # Apply feature embedding
        x = self.feature_embedding(x)

        # Reshape back to [batch, time, base_filters]
        x = x.reshape(batch_size, time_steps, -1)

        # Permute to [batch, base_filters, time] for conv1d operations
        x = x.permute(0, 2, 1)

        # Apply causal padding for initial conv
        x = F.pad(x, (1, 0))
        x = self.initial(x)

        # Process blocks and accumulate skip connections directly
        skip_sum = 0
        for block in self.blocks:
            residual = x

            # Causal padding
            padding = block['dilated_conv'].dilation[0] * (block['dilated_conv'].kernel_size[0] - 1)
            x = F.pad(x, (padding, 0))

            # Dilated conv and gating
            x = block['dilated_conv'](x)
            filter_x, gate_x = torch.chunk(x, 2, dim=1)
            x = torch.tanh(filter_x) * torch.sigmoid(gate_x)

            # Residual connection
            x = block['res_conv'](x)
            x = block['norm'](x)
            x = block['dropout'](x)
            x = x + residual

            # Add to skip connections sum directly
            skip_sum = skip_sum + x

        # Final processing
        x = self.global_pool(skip_sum)
        x = x.squeeze(-1)
        x = self.fc(x)

        x = F.softmax(x, dim=1)

        return x

    def random_init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def predict(self, x):
        self.eval()
        x = x.to('cuda:0')
        output = self(x)
        return output


class EnhancedResNet(nn.Module):
    def __init__(
            self,
            input_dim=11,  # Input feature dimension
            output_dim=2,  # Output dimension (usually 2: non-seizure/seizure)
            base_filters=32,  # Base filter count
            n_blocks=3,  # Number of residual blocks
            kernel_size=16,  # Convolution kernel size
            dropout=0.2,  # Dropout rate
            lr=0.001,  # Learning rate
            weight_decay=1e-5,  # Weight decay
            norm_type='batch',  # Normalization type
            gamma=0.5,  # Anatomical constraint loss weight
            delta=0.5  # Temporal consistency loss weight
    ):
        """
        Enhanced ResNet model with support for anatomical constraints and seizure channel masks
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.norm_layer = nn.BatchNorm1d if norm_type == 'batch' else nn.LayerNorm

        # Store loss weights
        self.gamma = gamma
        self.delta = delta

        # Initial feature embedding layer
        self.feature_embedding = nn.Linear(input_dim, base_filters)

        # Initial causal convolution
        self.initial = nn.Sequential(
            nn.Conv1d(base_filters, base_filters, kernel_size),
            self.norm_layer(base_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stack of dilated convolution blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 8)
            self.blocks.append(nn.ModuleDict({
                'dilated_conv': nn.Conv1d(
                    base_filters,
                    base_filters * 2,
                    kernel_size,
                    dilation=dilation,
                ),
                'norm': self.norm_layer(base_filters),
                'res_conv': nn.Conv1d(base_filters, base_filters, 1),
                'dropout': nn.Dropout(dropout)
            }))

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Create branches for different tasks
        hidden_size = base_filters

        # Main classification head - matches other models' output interface
        self.fc = nn.Linear(hidden_size, output_dim)

        # Set standard loss function and optimizer for compatibility
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Masks and weights
        self.seizure_mask = None
        self.grey_matter_values = None

        # Move to appropriate device
        self.to(self.device)

    def set_seizure_mask(self, mask):
        """Set seizure channel mask"""
        self.seizure_mask = mask

    def set_grey_matter_values(self, values):
        """Set grey matter values"""
        self.grey_matter_values = values

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input data (batch_size, channels, time_steps)

        Returns:
            Model output (batch_size, output_dim)
        """
        batch_size, n_features, time_steps = x.shape

        # Feature embedding
        x = x.permute(0, 2, 1).reshape(batch_size * time_steps, n_features)
        x = self.feature_embedding(x)
        x = x.reshape(batch_size, time_steps, -1).permute(0, 2, 1)

        # Apply causal padding for initial conv
        x = F.pad(x, (1, 0))
        x = self.initial(x)

        # Process blocks and accumulate skip connections
        skip_sum = 0
        for block in self.blocks:
            residual = x

            # Causal padding
            padding = block['dilated_conv'].dilation[0] * (block['dilated_conv'].kernel_size[0] - 1)
            x = F.pad(x, (padding, 0))

            # Dilated conv and gating
            x = block['dilated_conv'](x)
            filter_x, gate_x = torch.chunk(x, 2, dim=1)
            x = torch.tanh(filter_x) * torch.sigmoid(gate_x)

            # Residual connection
            x = block['res_conv'](x)
            x = block['norm'](x)
            x = block['dropout'](x)
            x = x + residual

            # Add to skip connections sum
            skip_sum = skip_sum + x

        # Global pooling to extract features
        features = self.global_pool(skip_sum).squeeze(-1)

        # Main output
        output = self.fc(features)

        return output

    def anatomical_constraint_loss(self, outputs, input_channel_indices):
        """
        Calculate anatomical constraint loss using grey matter values

        Args:
            outputs: Model outputs
            targets: Optional target labels
        """
        if self.grey_matter_values is None:
            return 0.0

        # Get seizure probabilities
        seizure_probs = F.softmax(outputs, dim=1)[:, 1]

        grey_matter_values = self.grey_matter_values

        # Calculate weighted loss based on grey matter values
        # Higher penalty for white matter (low grey_matter_values)
        # Lower penalty for grey matter (high grey_matter_values)
        white_matter_weight = 1.0 - grey_matter_values  # Invert values: 1.0 for white, 0.0 for grey

        # Apply threshold and calculate weighted loss
        threshold = 0.3  # Seizure probability threshold TODO: make this a hyperparameter
        weighted_loss = torch.clamp(seizure_probs * white_matter_weight - threshold, min=0)

        # Return the mean weighted loss
        return self.gamma * torch.mean(weighted_loss)

    def temporal_loss(self, outputs, input_channel_indices, input_time_indices):
        '''
        Enforce temporal consistency within the same channel
        '''
        seizure_probs = F.softmax(outputs, dim=1)[:, 1]

        loss = 0.0
        count = 0

        # Group by channel
        for ch in torch.unique(input_channel_indices):
            ch_mask = input_channel_indices == ch
            ch_times = input_time_indices[ch_mask]
            ch_probs = seizure_probs[ch_mask]

            # Sort by time
            sorted_times, sort_idx = torch.sort(ch_times)
            ch_probs_sorted = ch_probs[sort_idx]

            # Penalize if later prob < earlier prob
            diff = ch_probs_sorted[:-1] - ch_probs_sorted[1:]
            penalty = torch.clamp(diff, min=0)
            loss += torch.sum(penalty)
            count += len(penalty)

        if count == 0:
            return 0.0
        return self.delta * loss / count

    def custom_loss(self, outputs, targets, input_channel_indices, input_time_indices):
        """
        Combined loss function

        Args:
            outputs: Model outputs
            targets: Target labels

        Returns:
            Total loss value
        """
        # Standard classification loss
        classification_loss = self.criteria(outputs, targets)

        # Anatomical constraint loss
        anatomical_loss = self.anatomical_constraint_loss(outputs, input_channel_indices)

        # Seizure channel loss
        temporal_loss = self.temporal_loss(outputs, input_channel_indices, input_time_indices)

        # Combined loss
        total_loss = classification_loss + anatomical_loss + temporal_loss

        return total_loss

    def random_init(self):
        """Randomly initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_using_optimizer(
        model: Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        save_location: Optional[str] = None,
        epochs: int = 200,
        device: str = 'cuda:0',
        patience: int = 7,
        scheduler_patience: int = 5,
        checkpoint_freq: int = 20,
        gradient_clip: Optional[float] = None
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train a PyTorch model with comprehensive monitoring and checkpointing.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    trainloader : DataLoader
        Training data loader
    valloader : DataLoader
        Validation data loader
    save_location : str, optional
        Directory to save model checkpoints
    epochs : int
        Number of training epochs
    device : str
        Device to train on
    patience : int
        Early stopping patience
    scheduler_patience : int
        Learning rate scheduler patience
    checkpoint_freq : int
        Frequency of model checkpointing
    gradient_clip : float, optional
        Maximum gradient norm for gradient clipping

    Returns:
    --------
    Tuple containing:
        - Training losses
        - Validation losses
        - Validation accuracies
    """
    try:
        # Setup device
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        device = torch.device(device)
        model = model.to(device)

        # Initialize tracking variables
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []
        best_val_loss = float('inf')
        best_model_state = None

        # Initialize early stopping and learning rate scheduler
        early_stopping = EarlyStopping(patience=patience)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode='min', patience=scheduler_patience, verbose=True
        )

        # Initialize model
        try:
            model.random_init()
        except Exception as e:
            logger.warning(f"Model random initialization failed: {str(e)}")

        # Create save directory if needed
        if save_location:
            save_location = Path(save_location)
            save_location.mkdir(parents=True, exist_ok=True)

        # Training loop
        for epoch in range(epochs):
            try:
                # Training phase
                model.train()
                running_loss = 0.0
                batch_losses = []

                # Progress bar for training
                pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
                for i, (x, y) in enumerate(pbar):
                    try:
                        # Move data to device
                        x = x.float().to(device, non_blocking=True)
                        y = y.long().to(device, non_blocking=True)

                        # Zero gradients
                        model.optimizer.zero_grad()

                        # Forward pass
                        outputs = model(x).float()

                        # Compute loss
                        loss = model.criteria(outputs, y)
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss value at batch {i}")
                            continue

                        # Backward pass with gradient clipping
                        loss.backward()
                        if gradient_clip:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), gradient_clip
                            )

                        # Optimizer step
                        model.optimizer.step()

                        # Update metrics
                        batch_loss = loss.item()
                        running_loss += batch_loss
                        batch_losses.append(batch_loss)

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{batch_loss:.4f}',
                            'avg_loss': f'{running_loss / (i + 1):.4f}'
                        })

                    except Exception as e:
                        logger.error(f"Error in training batch {i}: {str(e)}")
                        continue

                # Calculate epoch metrics
                epoch_loss = running_loss / len(trainloader)
                train_losses.append(epoch_loss)

                # Validation phase
                if epoch % checkpoint_freq == checkpoint_freq - 1:
                    val_loss, val_accuracy = evaluate_model(model, valloader, 'cuda:0')
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)

                    # Update learning rate scheduler
                    scheduler.step(val_loss)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()

                    # Checkpointing
                    if save_location:
                        model_name = model.__class__.__name__
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': model.optimizer.state_dict(),
                            'train_loss': epoch_loss,
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy
                        }
                        save_file = save_location / f'{model_name}_epoch{epoch + 1}.pth'
                        torch.save(checkpoint, save_file)

                    # Print metrics
                    print(f'Epoch [{epoch + 1}/{epochs}]')
                    print(f'Training Loss: {epoch_loss:.4f}')
                    print(f'Validation Loss: {val_loss:.4f}')
                    print(f'Validation Accuracy: {val_accuracy:.4f}')

                    # Early stopping check
                    if early_stopping(val_loss):
                        print("Early stopping triggered")
                        break

            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
                continue

        # Save best model
        if save_location and best_model_state:
            best_model_path = save_location / f'{model_name}_best.pth'
            torch.save({
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss
            }, best_model_path)

        # Clear GPU memory
        torch.cuda.empty_cache()

        return train_losses, val_losses, val_accuracies

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def hyperparameter_search_for_model(
        model_name: str,
        model_creator: Callable,
        train_loader,
        val_loader,
        channels: int,
        time_steps: int,
        n_trials: int,
        search_space: dict,
        device: str,
        model_folder: str
) -> dict:
    """
    Perform hyperparameter search for a specific model using Optuna.

    Args:
        model_name: Name of the model
        model_creator: Function to create the model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        channels: Number of input channels
        time_steps: Number of time steps
        n_trials: Number of trials for hyperparameter search
        search_space: Dictionary defining the search space
        device: Device to use for training
        model_folder: Path to save study results

    Returns:
        dict: Dictionary of best hyperparameters
    """

    def objective(trial):
        # Create hyperparameters from search space
        hp = {}

        for param_name, config in search_space.items():
            if config['type'] == 'loguniform':
                hp[param_name] = trial.suggest_float(param_name, *config['range'], log=True)
            elif config['type'] == 'uniform':
                hp[param_name] = trial.suggest_float(param_name, *config['range'])
            elif config['type'] == 'int':
                hp[param_name] = trial.suggest_int(param_name, *config['range'])
            elif config['type'] == 'categorical':
                hp[param_name] = trial.suggest_categorical(param_name, config['values'])

        # Create the model
        model = model_creator(channels, time_steps, hp.get('lr', 0.001))

        # Apply dropout if the model supports it
        if 'dropout' in hp:
            try:
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = hp['dropout']
            except:
                pass  # Silently fail if dropout can't be set

        # Set gamma and delta for EnhancedResNet if they're in the hyperparameters
        if isinstance(model, EnhancedResNet):
            if 'gamma' in hp:
                model.gamma = hp['gamma']
            if 'delta' in hp:
                model.delta = hp['delta']

        # Mini training with early stopping for quick evaluation
        mini_epochs = min(15, n_trials)  # Reduced number of epochs for quick evaluation

        try:
            # Create a temporary directory for checkpoints
            temp_dir = os.path.join(model_folder, f"optuna_trial_{trial.number}")
            os.makedirs(temp_dir, exist_ok=True)

            # Choose the appropriate training function based on model type
            if isinstance(model, EnhancedResNet):
                # Train with limited epochs for hyperparameter search
                _, val_losses, val_accuracies = train_using_optimizer_with_masks(
                    model=model,
                    trainloader=train_loader,
                    valloader=val_loader,
                    save_location=temp_dir,
                    epochs=mini_epochs,
                    device=device,
                    patience=hp.get('patience', 5),
                    scheduler_patience=hp.get('scheduler_patience', 3),
                    gradient_clip=hp.get('gradient_clip', 1.0),
                    checkpoint_freq=1  # Evaluate at every epoch for optuna
                )
            else:
                # Use standard training for other models
                _, val_losses, val_accuracies = train_using_optimizer(
                    model=model,
                    trainloader=train_loader,
                    valloader=val_loader,
                    save_location=temp_dir,
                    epochs=mini_epochs,
                    device=device,
                    patience=hp.get('patience', 5),
                    scheduler_patience=hp.get('scheduler_patience', 3),
                    gradient_clip=hp.get('gradient_clip', 1.0),
                    checkpoint_freq=1  # Evaluate at every epoch for optuna
                )

            # Use the best validation accuracy as the objective
            trial.set_user_attr('val_losses', val_losses)

            if val_accuracies:
                best_acc = max(val_accuracies)
                return best_acc
            else:
                return 0.0  # In case no validation was performed

        except Exception as e:
            logger.error(f"Error in trial: {str(e)}")
            # Return a very poor score to avoid this configuration
            return 0.0

    # Create a study for hyperparameter optimization
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",  # Maximize validation accuracy
        pruner=pruner,
        sampler=sampler
    )

    # Run the optimization
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save visualization
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(model_folder, f"{model_name}_optimization_history.png"))

        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(model_folder, f"{model_name}_param_importances.png"))
    except Exception as e:
        logger.warning(f"Could not generate Optuna visualizations: {str(e)}")

    # Print results
    print(f"\nBest trial for {model_name}:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")

    return trial.params

def output_to_probability(model, x, device='cuda:0'):
    '''
    CAUTIOUS, THIS ONLY FITS FOR 2-CLASS AND 1 IS FOR SEIZURE CLASS
    This function output the probability of a given time-seris x to be a seizure
    :param model:
    :param x:
    :param device:
    :return:
    '''
    model.eval()
    x = x.to(device)
    output = model(x).detach().cpu().numpy()

    return output[:, 1]


def evaluate_model(
        model: Module,
        dataloader: DataLoader,
        device: str = 'cuda:0',
        num_batches: Optional[int] = None,
        return_predictions: bool = False
) -> Tuple[float, float]:
    """
    Evaluate a PyTorch model on a dataloader.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : torch.utils.data.DataLoader
        DataLoader containing validation/test data
    device : str
        Device to run evaluation on
    num_batches : Optional[int]
        Number of batches to evaluate (None for all batches)
    return_predictions : bool
        Whether to return predictions and targets

    Returns:
    --------
    Tuple containing:
        - Average loss
        - Average accuracy
        - (Optional) Predictions and targets if return_predictions=True
    """
    try:
        # Verify device availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'

        device = torch.device(device)
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        all_accuracies = []
        all_predictions = []
        all_targets = []
        total_samples = 0

        # Handle empty dataloader
        if len(dataloader) == 0:
            raise ValueError("DataLoader is empty")

        # Determine number of batches to evaluate
        if num_batches is None:
            num_batches = len(dataloader)
        elif num_batches > len(dataloader):
            logger.warning(
                f"num_batches ({num_batches}) is greater than available batches "
                f"({len(dataloader)}). Using all available batches."
            )
            num_batches = len(dataloader)

        with torch.no_grad():
            for i, (x, y) in enumerate(itertools.islice(dataloader, num_batches)):
                try:
                    # Move data to device
                    x = x.float().to(device, non_blocking=True)
                    y = y.long().to(device, non_blocking=True)

                    # Handle empty batch
                    if x.size(0) == 0:
                        continue

                    # Forward pass with error handling
                    try:
                        outputs = model(x).float()
                    except RuntimeError as e:
                        logger.error(f"Forward pass failed: {str(e)}")
                        raise

                    # Compute loss with numerical stability
                    try:
                        loss = model.criteria(outputs, y)
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss value at batch {i}")
                            continue
                    except Exception as e:
                        logger.error(f"Loss computation failed: {str(e)}")
                        raise

                    total_loss += loss.item() * x.size(0)

                    # Compute accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == y).float().mean().item()
                    all_accuracies.append(accuracy * x.size(0))

                    if return_predictions:
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(y.cpu().numpy())

                    total_samples += x.size(0)

                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    continue

        # Compute final metrics
        if total_samples == 0:
            raise ValueError("No valid samples processed")

        avg_loss = total_loss / total_samples
        avg_accuracy = sum(all_accuracies) / total_samples

        # Clear GPU memory
        torch.cuda.empty_cache()

        if return_predictions:
            return avg_loss, avg_accuracy, np.array(all_predictions), np.array(all_targets)
        return avg_loss, avg_accuracy

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def train_using_optimizer_with_masks(
        model,
        trainloader,
        valloader,
        save_location=None,
        epochs=200,
        device='cuda:0',
        patience=7,
        scheduler_patience=5,
        checkpoint_freq=20,
        gradient_clip=None
):
    """
    Enhanced training function with mask support

    Args:
        model: Neural network model
        trainloader: Training data loader with mask support
        valloader: Validation data loader with mask support
        save_location: Directory to save model checkpoints
        epochs: Number of training epochs
        device: Device to train on
        patience: Early stopping patience
        scheduler_patience: Learning rate scheduler patience
        checkpoint_freq: Frequency of model checkpointing
        gradient_clip: Maximum gradient norm for gradient clipping

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
    """
    # Setup device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)
    model = model.to(device)

    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_model_state = None

    # Initialize early stopping and learning rate scheduler
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', patience=scheduler_patience, verbose=True
    )

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        # Progress bar for training
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for i, batch in enumerate(pbar):
            # Move data to device
            data = batch['data'].float().to(device)
            labels = batch['label'].long().to(device)

            # Set masks if available
            if isinstance(model, EnhancedResNet):
                model.set_grey_matter_values(batch['grey_matter_values'].float().to(device))
                model.set_seizure_mask(batch['seizure_mask'].bool().to(device))
                input_channel_indices = batch['channel_idx'].long().to(device)
                input_time_indices = batch['time_idx'].long().to(device)

            # Forward pass
            outputs = model(data)

            # Compute loss
            if isinstance(model, EnhancedResNet) and hasattr(model, 'custom_loss'):
                loss = model.custom_loss(outputs, labels, input_channel_indices, input_time_indices)
            else:
                loss = model.criteria(outputs, labels)

            # Backward pass and gradient clipping
            model.optimizer.zero_grad()
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # Optimizer step
            model.optimizer.step()

            # Update metrics
            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (i + 1):.4f}'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)

        # Validation phase
        val_loss, val_accuracy = evaluate_model_with_masks(model, valloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Checkpointing
        if save_location and (epoch + 1) % checkpoint_freq == 0:
            model_name = model.__class__.__name__
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            save_file = os.path.join(save_location, f'{model_name}_epoch{epoch + 1}.pth')
            torch.save(checkpoint, save_file)

        # Print metrics
        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')

        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    # Save best model
    if save_location and best_model_state:
        model_name = model.__class__.__name__
        best_model_path = os.path.join(save_location, f'{model_name}_best.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'val_loss': best_val_loss
        }, best_model_path)

    # Clear GPU memory
    torch.cuda.empty_cache()

    return train_losses, val_losses, val_accuracies


def evaluate_model_with_masks(model, dataloader, device):
    """
    Evaluate model with mask support

    Args:
        model: Neural network model
        dataloader: Data loader with mask support
        device: Device to evaluate on

    Returns:
        avg_loss: Average loss
        avg_accuracy: Average accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            data = batch['data'].float().to(device)
            labels = batch['label'].long().to(device)

            # Set masks if available
            if isinstance(model, EnhancedResNet):
                model.set_grey_matter_values(batch['grey_matter_values'].float().to(device))
                model.set_seizure_mask(batch['seizure_mask'].bool().to(device))
                input_channel_indices = batch['channel_idx'].long().to(device)
                input_time_indices = batch['time_idx'].long().to(device)

            # Forward pass
            outputs = model(data)

            # Compute loss
            if isinstance(model, EnhancedResNet) and hasattr(model, 'custom_loss'):
                loss = model.custom_loss(outputs, labels, input_channel_indices, input_time_indices)
            else:
                loss = model.criteria(outputs, labels)

            total_loss += loss.item() * data.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average metrics
    avg_loss = total_loss / total
    avg_accuracy = correct / total

    return avg_loss, avg_accuracy


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention is All You Need'."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) Concatenate and apply final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed-forward."""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Stack of N encoder layers."""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class CustomTransformerSeizurePredictor(BaseModel):
    """Transformer for seizure channel prediction (encoder only, no decoder)."""
    def __init__(self, input_dim, output_dim=2, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_seq_len=1000, lr=0.0001, weight_decay=1e-5, device='cuda:0'):
        super().__init__(input_dim, output_dim, device)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        attn = MultiHeadedAttention(n_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, attn, ff, dropout)
        self.encoder = Encoder(encoder_layer, n_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.to(self.device)
        # If input is (batch, channels, seq_len), transpose to (batch, seq_len, channels)
        if len(x.shape) == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, d_model)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

    def random_init(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def detect_model_type(model):
    """
    检测模型类型以确定使用哪种损失函数和评估方法
    """
    model_name = model.__class__.__name__
    if 'Transformer' in model_name or 'CustomTransformer' in model_name:
        return 'transformer'
    else:
        return 'standard'


def universal_evaluate_model(model, dataloader, device='cuda', model_type=None):
    """
    通用模型评估函数，兼容所有模型类型

    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 设备
        model_type: 模型类型 ('transformer' 或 'standard')

    Returns:
        avg_loss: 平均损失
        avg_accuracy: 平均准确率
    """
    if model_type is None:
        model_type = detect_model_type(model)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 根据模型类型选择损失函数
    if model_type == 'transformer':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_data in dataloader:
            try:
                # 处理不同的数据格式
                if isinstance(batch_data, dict):
                    # 字典格式 (如EnhancedResNet)
                    data = batch_data['data'].float().to(device)
                    labels = batch_data['label'].long().to(device)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    # 元组格式 (标准格式)
                    data, labels = batch_data
                    data = data.float().to(device)
                    labels = labels.to(device)
                else:
                    # 直接是数据
                    data = batch_data.float().to(device)
                    labels = None

                # 前向传播
                outputs = model(data)

                if labels is not None:
                    if model_type == 'transformer':
                        # Transformer使用KL散度损失
                        if labels.dtype == torch.long:
                            # 如果标签是long类型，转换为概率分布
                            labels_one_hot = torch.zeros_like(outputs)
                            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                            labels = labels_one_hot.float()

                        # 确保输出是有效的概率分布
                        outputs = torch.clamp(outputs, min=1e-8, max=1.0)
                        outputs = outputs / outputs.sum(dim=1, keepdim=True)
                        log_outputs = torch.log(outputs)

                        loss = criterion(log_outputs, labels)
                    else:
                        # 标准模型使用交叉熵损失
                        if labels.dtype == torch.float and labels.shape[1] > 1:
                            # 概率分布标签转换为类别标签
                            labels = torch.argmax(labels, dim=1)
                        loss = criterion(outputs, labels)

                    total_loss += loss.item()

                    # 计算准确率
                    if model_type == 'transformer' and labels.shape[1] > 1:
                        # Transformer的概率分布标签
                        predicted = torch.argmax(outputs, dim=1)
                        true_labels = torch.argmax(labels, dim=1)
                    else:
                        # 标准标签
                        predicted = torch.argmax(outputs, dim=1)
                        true_labels = labels

                    total += labels.size(0)
                    correct += (predicted == true_labels).sum().item()

            except Exception as e:
                logger.warning(f"Error in evaluation batch: {str(e)}")
                continue

    if total > 0:
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = correct / total
        return avg_loss, avg_accuracy
    else:
        return float('inf'), 0.0


def universal_train_model(
        model,
        train_loader,
        val_loader,
        save_location: Optional[str] = None,
        epochs: int = 50,
        device: str = 'cuda',
        patience: int = 7,
        scheduler_patience: int = 5,
        gradient_clip: Optional[float] = 1.0,
        checkpoint_freq: int = 10,
        trial: Optional[optuna.Trial] = None,
        model_type: Optional[str] = None
) -> Tuple[List[float], List[float], List[float]]:
    """
    通用训练函数，兼容所有模型类型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        save_location: 模型保存位置
        epochs: 训练轮数
        device: 设备
        patience: 早停耐心值
        scheduler_patience: 学习率调度器耐心值
        gradient_clip: 梯度裁剪
        checkpoint_freq: 检查点保存频率
        trial: Optuna试验对象（用于剪枝）
        model_type: 模型类型

    Returns:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
    """
    # 检测模型类型
    if model_type is None:
        model_type = detect_model_type(model)

    # 设置设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA不可用，回退到CPU")
        device = 'cpu'
    device = torch.device(device)
    model = model.to(device)

    # 获取优化器（优先使用模型自带的）
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        optimizer = model.optimizer
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 根据模型类型选择损失函数
    if model_type == 'transformer':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = nn.CrossEntropyLoss()

    # 初始化跟踪变量
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    # 早停和学习率调度器
    early_stopping = EarlyStopping(patience=patience)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience, verbose=True
    )

    # 创建保存目录
    if save_location:
        save_location = Path(save_location)
        save_location.mkdir(parents=True, exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        num_batches = 0

        # 是否显示进度条（Optuna试验时禁用）
        disable_pbar = trial is not None
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}',
                    disable=disable_pbar, leave=False)

        for batch_idx, batch_data in enumerate(pbar):
            try:
                # 处理不同的数据格式
                if isinstance(batch_data, dict):
                    # 字典格式
                    data = batch_data['data'].float().to(device)
                    labels = batch_data['label'].to(device)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    # 元组格式
                    data, labels = batch_data
                    data = data.float().to(device)
                    labels = labels.to(device)
                else:
                    logger.warning(f"未知的批次数据格式: {type(batch_data)}")
                    continue

                optimizer.zero_grad()

                # 前向传播
                outputs = model(data)

                # 计算损失
                if model_type == 'transformer':
                    # Transformer使用KL散度损失
                    if labels.dtype == torch.long:
                        # 转换为概率分布
                        labels_one_hot = torch.zeros_like(outputs)
                        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                        labels = labels_one_hot.float()

                    # 确保输出是有效的概率分布
                    outputs = torch.clamp(outputs, min=1e-8, max=1.0)
                    outputs = outputs / outputs.sum(dim=1, keepdim=True)
                    log_outputs = torch.log(outputs)

                    loss = criterion(log_outputs, labels)
                else:
                    # 标准模型使用交叉熵损失
                    if labels.dtype == torch.float and labels.shape[1] > 1:
                        # 概率分布标签转换为类别标签
                        labels = torch.argmax(labels, dim=1)
                    loss = criterion(outputs, labels)

                # 检查数值稳定性
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"训练第{epoch + 1}轮第{batch_idx}批次出现异常损失值")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # 更新进度条
                if not disable_pbar:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss / num_batches:.4f}'
                    })

            except Exception as e:
                logger.error(f"训练第{epoch + 1}轮第{batch_idx}批次出错: {str(e)}")
                continue

        # 计算平均训练损失
        if num_batches > 0:
            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
        else:
            logger.warning(f"第{epoch + 1}轮没有有效的训练批次")
            continue

        # 验证阶段
        val_loss, val_acc = universal_evaluate_model(model, val_loader, device, model_type)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_location:
                save_model_checkpoint(
                    model, optimizer, scheduler, epoch,
                    avg_train_loss, val_loss, val_acc,
                    save_location / 'best_model.pth'
                )

        # 定期保存检查点
        if save_location and (epoch + 1) % checkpoint_freq == 0:
            model_name = model.__class__.__name__
            checkpoint_path = save_location / f'{model_name}_epoch{epoch + 1}.pth'
            save_model_checkpoint(
                model, optimizer, scheduler, epoch,
                avg_train_loss, val_loss, val_acc, checkpoint_path
            )

        # 打印指标
        if trial is None:
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {val_acc:.4f}')
        else:
            # Optuna试验中的简化输出
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}: Val Acc: {val_acc:.4f}')

        # Optuna剪枝
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"Trial在第{epoch + 1}轮被剪枝")
                raise optuna.exceptions.TrialPruned()

        # 早停检查
        if early_stopping(val_loss):
            print(f"第{epoch + 1}轮触发早停")
            break

    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return train_losses, val_losses, val_accuracies


def save_model_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, val_acc, save_path):
    """保存模型检查点（带完整 config）"""
    # 1) 优先用模型自带配置
    if hasattr(model, "get_config") and callable(model.get_config):
        model_config = model.get_config()
    elif hasattr(model, "init_kwargs"):  # 你可以在模型 __init__ 最后加上 self.init_kwargs = locals()
        model_config = dict(model.init_kwargs)
        model_config.pop("self", None)  # 去掉 self
    else:
        model_config = {}

    # 2) 确保写入 input_dim / output_dim
    state = model.state_dict()
    if "input_dim" not in model_config or model_config["input_dim"] is None:
        for k, v in state.items():
            if k.endswith("weight") and v.ndim == 2:  # Linear
                model_config["input_dim"] = int(v.shape[1])
                break
    if "output_dim" not in model_config or model_config["output_dim"] is None:
        for k, v in state.items():
            if k.endswith("weight") and v.ndim == 2:  # Linear
                model_config["output_dim"] = int(v.shape[0])
        # 某些模型用 classifier/conv 末层
        if "output_dim" not in model_config:
            for k, v in state.items():
                if ("classifier" in k or "fc" in k) and k.endswith("weight"):
                    model_config["output_dim"] = int(v.shape[0])
                    break

    # 3) 组装 checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "model_config": model_config,
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_path)
    print(f"✅ 模型保存到 {save_path}")
    print(f"📋 model_config: {model_config}")



def save_trial_details_to_txt(
        trial_number: int,
        model_name: str,
        params: Dict[str, Any],
        train_losses: List[float],
        val_losses: List[float],
        val_accuracies: List[float],
        best_accuracy: float,
        training_time: Optional[float] = None,
        save_folder: str = "trial_logs",
        additional_info: Optional[Dict] = None
):
    """
    保存每次trial的训练细节到txt文件

    Args:
        trial_number: Trial编号
        model_name: 模型名称
        params: 超参数字典
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
        best_accuracy: 最佳准确率
        training_time: 训练耗时（秒）
        save_folder: 保存文件夹
        additional_info: 额外信息字典
    """

    # 确保保存文件夹存在
    os.makedirs(save_folder, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trial_{trial_number:03d}_{model_name}_{timestamp}.txt"
    filepath = os.path.join(save_folder, filename)

    # 计算统计信息
    final_train_loss = train_losses[-1] if train_losses else "N/A"
    final_val_loss = val_losses[-1] if val_losses else "N/A"
    final_val_acc = val_accuracies[-1] if val_accuracies else "N/A"
    epochs_trained = len(train_losses)

    # 找到最佳epoch
    if val_accuracies:
        best_epoch = val_accuracies.index(max(val_accuracies)) + 1
        best_val_loss = val_losses[best_epoch - 1] if val_losses else "N/A"
    else:
        best_epoch = "N/A"
        best_val_loss = "N/A"

    # 计算训练改善情况
    if len(train_losses) >= 2:
        train_improvement = train_losses[0] - train_losses[-1]
        train_improvement_pct = (train_improvement / train_losses[0]) * 100
    else:
        train_improvement = "N/A"
        train_improvement_pct = "N/A"

    # 写入txt文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"OPTUNA TRIAL #{trial_number} - {model_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"Trial编号: {trial_number}\n")
        if training_time:
            f.write(f"训练耗时: {training_time:.2f} 秒 ({training_time / 60:.1f} 分钟)\n")
        f.write("\n")

        # 超参数部分
        f.write("🔧 超参数配置:\n")
        f.write("-" * 30 + "\n")
        for key, value in params.items():
            if isinstance(value, float):
                if value < 0.001:
                    f.write(f"  {key:15}: {value:.2e}\n")
                else:
                    f.write(f"  {key:15}: {value:.4f}\n")
            else:
                f.write(f"  {key:15}: {value}\n")
        f.write("\n")

        # 训练结果概览
        f.write("📊 训练结果概览:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  训练轮数:     {epochs_trained}\n")
        f.write(f"  最佳准确率:   {best_accuracy:.4f} (第{best_epoch}轮)\n")
        f.write(f"  最佳时验证损失: {best_val_loss}\n")
        f.write(f"  最终训练损失: {final_train_loss}\n")
        f.write(f"  最终验证损失: {final_val_loss}\n")
        f.write(f"  最终验证准确率: {final_val_acc}\n")
        if train_improvement != "N/A":
            f.write(f"  训练损失改善: {train_improvement:.4f} ({train_improvement_pct:.1f}%)\n")
        f.write("\n")

        # 详细训练历史
        f.write("📈 详细训练历史:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12} {'Notes'}\n")
        f.write("-" * 50 + "\n")

        for i in range(epochs_trained):
            epoch = i + 1
            train_loss = train_losses[i] if i < len(train_losses) else "N/A"
            val_loss = val_losses[i] if i < len(val_losses) else "N/A"
            val_acc = val_accuracies[i] if i < len(val_accuracies) else "N/A"

            # 添加注释
            notes = ""
            if val_accuracies and i < len(val_accuracies):
                if val_accuracies[i] == max(val_accuracies):
                    notes += "🏆 BEST"
                elif i > 0 and val_accuracies[i] > val_accuracies[i - 1]:
                    notes += "📈 UP"
                elif i > 0 and val_accuracies[i] < val_accuracies[i - 1]:
                    notes += "📉 DOWN"

            # 格式化数值
            if isinstance(train_loss, float):
                train_str = f"{train_loss:.4f}"
            else:
                train_str = str(train_loss)

            if isinstance(val_loss, float):
                val_loss_str = f"{val_loss:.4f}"
            else:
                val_loss_str = str(val_loss)

            if isinstance(val_acc, float):
                val_acc_str = f"{val_acc:.4f}"
            else:
                val_acc_str = str(val_acc)

            f.write(f"{epoch:<6} {train_str:<12} {val_loss_str:<12} {val_acc_str:<12} {notes}\n")

        f.write("\n")

        # 额外信息
        if additional_info:
            f.write("ℹ️  额外信息:\n")
            f.write("-" * 20 + "\n")
            for key, value in additional_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # 原始数据（便于后续分析）
        f.write("📋 原始数据 (JSON格式):\n")
        f.write("-" * 30 + "\n")
        raw_data = {
            "trial_number": trial_number,
            "model_name": model_name,
            "timestamp": timestamp,
            "params": params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_accuracy": best_accuracy,
            "epochs_trained": epochs_trained,
            "training_time": training_time
        }
        f.write(json.dumps(raw_data, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("Trial记录结束\n")
        f.write("=" * 70 + "\n")

    print(f"📝 Trial #{trial_number} 详细记录已保存到: {filepath}")
    return filepath


def create_trial_summary(save_folder: str = "trial_logs"):
    """
    创建所有trials的汇总文件

    Args:
        save_folder: trial日志文件夹
    """
    if not os.path.exists(save_folder):
        print(f"文件夹 {save_folder} 不存在")
        return

    # 查找所有trial文件
    trial_files = [f for f in os.listdir(save_folder) if f.startswith("trial_") and f.endswith(".txt")]

    if not trial_files:
        print("没有找到trial文件")
        return

    # 排序文件
    trial_files.sort()

    # 创建汇总文件
    summary_path = os.path.join(save_folder, "trials_summary.txt")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("OPTUNA TRIALS 汇总报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总trial数: {len(trial_files)}\n")
        f.write("\n")

        f.write("📊 所有Trials概览:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Trial':<6} {'Model':<20} {'Best Acc':<10} {'Batch Size':<12} {'LR':<12} {'Status'}\n")
        f.write("-" * 80 + "\n")

        # 解析每个trial文件
        best_overall = {"trial": 0, "accuracy": 0, "model": ""}

        for file in trial_files:
            try:
                with open(os.path.join(save_folder, file), 'r', encoding='utf-8') as trial_file:
                    content = trial_file.read()

                    # 提取关键信息
                    lines = content.split('\n')
                    trial_num = "N/A"
                    model_name = "N/A"
                    best_acc = 0
                    batch_size = "N/A"
                    lr = "N/A"

                    for line in lines:
                        if "Trial编号:" in line:
                            trial_num = line.split(":")[-1].strip()
                        elif "模型:" in line and "Trial编号" not in line:
                            model_name = line.split(":")[-1].strip()
                        elif "最佳准确率:" in line:
                            try:
                                best_acc = float(line.split(":")[1].split("(")[0].strip())
                            except:
                                best_acc = 0
                        elif "batch_size" in line and ":" in line:
                            batch_size = line.split(":")[-1].strip()
                        elif "lr" in line and ":" in line:
                            try:
                                lr_val = float(line.split(":")[-1].strip())
                                lr = f"{lr_val:.2e}"
                            except:
                                lr = line.split(":")[-1].strip()

                    # 记录最佳结果
                    if best_acc > best_overall["accuracy"]:
                        best_overall = {"trial": trial_num, "accuracy": best_acc, "model": model_name}

                    # 状态判断
                    status = "完成"
                    if best_acc == 0:
                        status = "失败"
                    elif best_acc > 0.9:
                        status = "优秀"
                    elif best_acc > 0.8:
                        status = "良好"

                    f.write(f"{trial_num:<6} {model_name:<20} {best_acc:<10.4f} {batch_size:<12} {lr:<12} {status}\n")

            except Exception as e:
                f.write(f"解析 {file} 时出错: {str(e)}\n")

        f.write("\n")
        f.write("🏆 最佳结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trial #{best_overall['trial']} - {best_overall['model']}\n")
        f.write(f"最佳准确率: {best_overall['accuracy']:.4f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("汇总报告结束\n")
        f.write("=" * 80 + "\n")

    print(f"📊 Trials汇总报告已保存到: {summary_path}")
    return summary_path


def universal_hyperparameter_search(
        model_class,
        train_loader,
        val_loader,
        input_dim: int,
        output_dim: int = 2,
        n_trials: int = 20,
        device: str = 'cuda',
        model_folder: str = 'checkpoints',
        search_space: Optional[Dict] = None
) -> Tuple[Dict, Any]:
    """
    通用超参数搜索函数，兼容所有模型类型

    Args:
        model_class: 模型类
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        input_dim: 输入维度
        output_dim: 输出维度
        n_trials: 试验次数
        device: 设备
        model_folder: 模型保存文件夹
        search_space: 搜索空间配置

    Returns:
        best_params: 最佳参数
        study: Optuna研究对象
    """

    def objective(trial):
        # 根据模型类型定义搜索空间
        model_name = model_class.__name__

        if 'Transformer' in model_name or 'CustomTransformer' in model_name:
            # Transformer特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
                'd_model': trial.suggest_categorical('d_model', [128, 256, 512, 1024]),
                'n_heads': trial.suggest_categorical('n_heads', [8, 16, 32]),
                'n_layers': trial.suggest_categorical('n_layers', [2, 3, 4, 6]),
                'd_ff': trial.suggest_categorical('d_ff', [512, 1024, 2048]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            }
        elif 'ResNet' in model_name:
            # ResNet特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'base_filters': trial.suggest_categorical('base_filters', [16, 32, 64, 128]),
                'kernel_size': trial.suggest_categorical('kernel_size', [8, 16, 32]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'n_blocks': trial.suggest_int('n_blocks', 2, 6),
            }
        elif 'CNN1D' in model_name:
            # CNN1D特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                'kernel_size': trial.suggest_categorical('kernel_size', [8, 16, 32]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
        elif 'Wavenet' in model_name:
            # Wavenet特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                'kernel_size': trial.suggest_categorical('kernel_size', [8, 16, 32]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
        elif 'LSTM' in model_name:
            # LSTM特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 50, 100, 200]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
        elif 'S4' in model_name:
            # S4特定参数
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128, 256]),
                'n_layers': trial.suggest_int('n_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
        else:
            # 默认搜索空间
            params = {
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }

        # 如果提供了自定义搜索空间，则覆盖默认值
        if search_space:
            for param_name, config in search_space.items():
                if config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, *config['range'], log=True)
                elif config['type'] == 'uniform':
                    params[param_name] = trial.suggest_float(param_name, *config['range'])
                elif config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, *config['range'])
                elif config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, config['values'])

        try:
            # 创建模型
            if 'Transformer' in model_name or 'CustomTransformer' in model_name:
                model = model_class(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    d_model=params.get('d_model', 512),
                    n_heads=params.get('n_heads', 8),
                    n_layers=params.get('n_layers', 6),
                    d_ff=params.get('d_ff', 2048),
                    dropout=params.get('dropout', 0.1),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5),
                    device=device
                )
            elif 'ResNet' in model_name:
                model = model_class(
                    input_dim=input_dim,
                    n_classes=output_dim,
                    base_filters=params.get('base_filters', 32),
                    kernel_size=params.get('kernel_size', 16),
                    dropout=params.get('dropout', 0.2),
                    n_blocks=params.get('n_blocks', 3),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5)
                )
            elif 'CNN1D' in model_name:
                model = model_class(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=params.get('hidden_dim', 128),
                    kernel_size=params.get('kernel_size', 256),
                    dropout=params.get('dropout', 0.2),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5),
                    device=device
                )
            elif 'Wavenet' in model_name:
                model = model_class(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=params.get('hidden_dim', 128),
                    kernel_size=params.get('kernel_size', 32),
                    dropout=params.get('dropout', 0.2),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5)
                )
            elif 'LSTM' in model_name:
                model = model_class(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=params.get('hidden_dim', 50),
                    num_layers=params.get('num_layers', 1),
                    dropout=params.get('dropout', 0.2),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5),
                    device=device
                )
            elif 'S4' in model_name:
                model = model_class(
                    d_input=input_dim,
                    d_output=output_dim,
                    d_model=params.get('d_model', 64),
                    n_layers=params.get('n_layers', 1),
                    dropout=params.get('dropout', 0.2),
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 1e-5)
                )
            else:
                # 通用模型创建
                model = model_class(input_dim=input_dim, output_dim=output_dim, lr=params['lr'])

            # 创建临时保存目录
            temp_dir = os.path.join(model_folder, f"optuna_trial_{trial.number}")
            os.makedirs(temp_dir, exist_ok=True)

            # 保存试验配置
            trial_config = {
                'trial_number': trial.number,
                'model_class': model_name,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }

            config_path = os.path.join(temp_dir, 'trial_config.json')
            with open(config_path, 'w') as f:
                json.dump(trial_config, f, indent=2)

            # 训练模型
            train_losses, val_losses, val_accuracies = universal_train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                save_location=temp_dir,
                epochs=15,  # 为快速搜索使用较少的轮数
                device=device,
                patience=5,
                trial=trial  # 传递trial对象用于剪枝
            )

            # 返回最佳验证准确率
            if val_accuracies:
                best_acc = max(val_accuracies)

                # 保存试验结果
                trial_config['results'] = {
                    'best_val_accuracy': best_acc,
                    'final_train_loss': train_losses[-1] if train_losses else None,
                    'final_val_loss': val_losses[-1] if val_losses else None,
                    'converged_epochs': len(val_accuracies)
                }

                with open(config_path, 'w') as f:
                    json.dump(trial_config, f, indent=2)

                save_trial_details_to_txt(
                    trial_number=trial.number,
                    model_name=model_name,
                    params=params,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    val_accuracies=val_accuracies,
                    best_accuracy=best_acc,
                    save_folder=os.path.join(model_folder, "trial_logs"),
                    additional_info={
                        "converged_epochs": len(val_accuracies),
                        "total_trials": n_trials,
                        "device": device
                    }
                )

                print(f"Trial {trial.number} completed with accuracy: {best_acc:.4f}")
                return best_acc
            else:
                print(f"Trial {trial.number} failed: No validation accuracies recorded")
                return 0.0

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return 0.0

    # 创建保存目录
    os.makedirs(model_folder, exist_ok=True)

    # 创建Optuna study
    study_name = f"{model_class.__name__}_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )

    print(f"开始{model_class.__name__}超参数搜索，共{n_trials}次试验...")

    # 运行优化
    study.optimize(objective, n_trials=n_trials)

    # 保存study结果
    study_results_path = os.path.join(model_folder, f"{study_name}_results.json")
    study_summary = {
        'study_name': study_name,
        'model_class': model_class.__name__,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'timestamp': datetime.now().isoformat(),
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }

    with open(study_results_path, 'w') as f:
        json.dump(study_summary, f, indent=2)

    create_trial_summary(save_folder=os.path.join(model_folder, "trial_logs"))

    # 打印最佳结果
    print(f"\n" + "=" * 60)
    print(f"{model_class.__name__}超参数搜索完成!")
    print(f"=" * 60)
    print(f"最佳试验 #{study.best_trial.number}:")
    print(f"  验证准确率: {study.best_value:.4f}")
    print(f"  最佳参数:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"  详细结果保存在: {study_results_path}")
    print(f"=" * 60)

    return study.best_params, study


def create_model_with_best_params(model_class, best_params, input_dim, output_dim=2, device='cuda'):
    """
    使用最佳参数创建模型

    Args:
        model_class: 模型类
        best_params: 最佳参数字典
        input_dim: 输入维度
        output_dim: 输出维度
        device: 设备

    Returns:
        model: 创建的模型实例
    """
    model_name = model_class.__name__

    try:
        if 'Transformer' in model_name or 'CustomTransformer' in model_name:
            model = model_class(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=best_params.get('d_model', 512),
                n_heads=best_params.get('n_heads', 8),
                n_layers=best_params.get('n_layers', 6),
                d_ff=best_params.get('d_ff', 2048),
                dropout=best_params.get('dropout', 0.1),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5),
                device=device
            )
        elif 'ResNet' in model_name:
            model = model_class(
                input_dim=input_dim,
                n_classes=output_dim,
                base_filters=best_params.get('base_filters', 32),
                kernel_size=best_params.get('kernel_size', 16),
                dropout=best_params.get('dropout', 0.2),
                n_blocks=best_params.get('n_blocks', 3),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5)
            )
        elif 'CNN1D' in model_name:
            model = model_class(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=best_params.get('hidden_dim', 128),
                kernel_size=best_params.get('kernel_size', 256),
                dropout=best_params.get('dropout', 0.2),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5),
                device=device
            )
        elif 'Wavenet' in model_name:
            model = model_class(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=best_params.get('hidden_dim', 128),
                kernel_size=best_params.get('kernel_size', 32),
                dropout=best_params.get('dropout', 0.2),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5)
            )
        elif 'LSTM' in model_name:
            model = model_class(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=best_params.get('hidden_dim', 50),
                num_layers=best_params.get('num_layers', 1),
                dropout=best_params.get('dropout', 0.2),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5),
                device=device
            )
        elif 'S4' in model_name:
            model = model_class(
                d_input=input_dim,
                d_output=output_dim,
                d_model=best_params.get('d_model', 64),
                n_layers=best_params.get('n_layers', 1),
                dropout=best_params.get('dropout', 0.2),
                lr=best_params.get('lr', 0.001),
                weight_decay=best_params.get('weight_decay', 1e-5)
            )
        else:
            # 通用模型创建
            model = model_class(
                input_dim=input_dim,
                output_dim=output_dim,
                lr=best_params.get('lr', 0.001)
            )

        print(f"✅ 成功创建{model_name}模型，使用最佳参数")
        return model

    except Exception as e:
        print(f"❌ 创建{model_name}模型失败: {str(e)}")
        return None


def load_model_with_config(checkpoint_path, model_class):
    """
    load model from checkpoints with config

    Args:
        checkpoint_path: checkpoint path
        model_class: model class

    Returns:
        model: loaded model
        checkpoint: checkpoints result
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_config' not in checkpoint:
        raise ValueError(f"checkpoint {checkpoint_path} is missing config info")

    # load model config
    config = checkpoint['model_config']

    # create model based on config
    model_name = model_class.__name__

    if 'Transformer' in model_name or 'CustomTransformer' in model_name:
        model = model_class(
            input_dim=config.get('input_dim'),
            output_dim=config.get('output_dim', 2),
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6),
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 1000)
        )
    elif 'ResNet' in model_name:
        model = model_class(
            input_dim=config.get('input_dim'),
            n_classes=config.get('output_dim', 2),
            base_filters=config.get('base_filters', 32),
            kernel_size=config.get('kernel_size', 16),
            n_blocks=config.get('n_blocks', 3),
            dropout=config.get('dropout', 0.2)
        )
    elif 'CNN1D' in model_name:
        model = model_class(
            input_dim=config.get('input_dim'),
            output_dim=config.get('output_dim', 2),
            hidden_dim=config.get('hidden_dim', 128),
            kernel_size=config.get('kernel_size', 256),
            dropout=config.get('dropout', 0.2)
        )
    elif 'Wavenet' in model_name:
        model = model_class(
            input_dim=config.get('input_dim'),
            output_dim=config.get('output_dim', 2),
            hidden_dim=config.get('hidden_dim', 128),
            kernel_size=config.get('kernel_size', 32),
            dropout=config.get('dropout', 0.2)
        )
    elif 'LSTM' in model_name:
        model = model_class(
            input_dim=config.get('input_dim'),
            output_dim=config.get('output_dim', 2),
            hidden_dim=config.get('hidden_dim', 50),
            num_layers=config.get('num_layers', 1),
            dropout=config.get('dropout', 0.2)
        )
    elif 'S4' in model_name:
        model = model_class(
            d_input=config.get('input_dim'),
            d_output=config.get('output_dim', 2),
            d_model=config.get('d_model', 64),
            n_layers=config.get('n_layers', 1),
            dropout=config.get('dropout', 0.2)
        )
    else:
        # general model
        model = model_class(
            input_dim=config.get('input_dim'),
            output_dim=config.get('output_dim', 2)
        )

    # load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Successfully Load{model_name}")
    print(f"📋 Model Config: {config}")
    if 'timestamp' in checkpoint:
        print(f"🕐 Save time: {checkpoint['timestamp']}")

    return model, checkpoint
