import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import itertools
from typing import List, Tuple, Callable, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import logging
from s4 import S4Block
from s4d import S4D
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import os

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
