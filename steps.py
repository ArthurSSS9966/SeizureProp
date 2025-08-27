import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, decimate
from sklearn.preprocessing import RobustScaler
from scipy import signal
from scipy.fftpack import fft
from statsmodels.tsa.ar_model import AutoReg
from meegkit import dss
from tqdm import tqdm
import pickle
from typing import Tuple, Optional, List, Dict
import pandas as pd
import logging
import torch
import json
import datetime
import torch.nn as nn

from utils import (butter_bandpass_filter,split_data, map_seizure_channels,
                   find_seizure_related_channels, map_channels_to_numbers)
from plotFun import plot_time_limited_heatmap, plot_eeg_style
from datasetConstruct import (combine_loaders,
                              load_seizure_across_patients, create_dataset,
                              EDFData, load_single_seizure)
from models import (train_using_optimizer, evaluate_model, output_to_probability,
                    Wavenet, CNN1D,LSTM, S4Model, ResNet, EnhancedResNet,
                    hyperparameter_search_for_model, train_using_optimizer_with_masks)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    """Class for handling EEG data processing and visualization."""

    def __init__(self, sampling_rate: int = 2000):
        self.sampling_rate = sampling_rate

    def plot_eeg_data(self, data: np.ndarray, timerange: np.ndarray,
                      title: str, save_path: Optional[str] = None,
                      ylim: Tuple[float, float] = (-400, 400)) -> None:
        """Plot EEG data with consistent formatting."""
        plt.figure(figsize=(12, 6))
        plt.plot(timerange, data)
        plt.vlines(0, ylim[0], ylim[1], color='r', linestyles='dashed',
                   label='Seizure Start')
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def compute_psd(self, data: np.ndarray, title: str,
                    save_path: Optional[str] = None) -> None:
        """Compute and plot power spectral density."""
        nperseg = self.sampling_rate // 2
        noverlap = nperseg // 4
        f, Pxx = welch(data, fs=self.sampling_rate, nperseg=nperseg,
                       noverlap=noverlap)

        plt.figure(figsize=(10, 6))
        plt.plot(f, Pxx)
        plt.title(title)
        plt.xlim(0, 100)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()


def init_examination(dataset, channel_number: int, result_folder: str,
                     start_time: float = -3, end_time: float = 7) -> None:
    """
    Initialize examination of seizure data with comprehensive visualization.

    Args:
        dataset: sEEG .edf dataset
        channel_number: Channel number to examine
        result_folder: Folder to save results
        start_time: Start time of examination window
        end_time: End time of examination window
    """
    try:
        os.makedirs(result_folder, exist_ok=True)
        processor = EEGProcessor(int(dataset.info["sfreq"]))

        # Extract data
        eeg_data, _ = dataset[:]
        eeg_data = eeg_data.T * 1e6
        channel_name = dataset.ch_names[channel_number]

        # Calculate indices
        time_start_idx = int(dataset.annotations.onset[
                                 np.where(dataset.annotations.description == 'SZ')[0][0]] *
                             processor.sampling_rate)
        time_start_idx = int(time_start_idx + start_time * processor.sampling_rate)
        time_end_idx = int(time_start_idx + end_time * processor.sampling_rate)

        # Extract channels
        channel1 = eeg_data[time_start_idx:time_end_idx, channel_number]
        channel2 = eeg_data[time_start_idx:time_end_idx, channel_number + 1]
        timerange = np.linspace(start_time - 1, end_time - 1, len(channel1))

        # Process and plot data
        plots = [
            ("raw", channel1),
            ("filtered", process_channel(channel1, processor.sampling_rate)),
            ("bipolar", channel1 - channel2),
            ("bipolar_filtered", process_channel(channel1 - channel2,
                                                 processor.sampling_rate))
        ]

        for plot_type, data in plots:
            save_path = os.path.join(
                result_folder,
                f"seizure_{dataset.seizureNumber}_{channel_name}_{plot_type}.svg"
            )
            processor.plot_eeg_data(
                data, timerange,
                f"Seizure {dataset.seizureNumber}_Channel {channel_name} {plot_type.title()}",
                save_path
            )

            # Compute PSD for each type
            psd_save_path = os.path.join(
                result_folder,
                f"seizure_{dataset.seizureNumber}_{plot_type}_psd.png"
            )
            processor.compute_psd(
                data,
                f"Seizure {dataset.seizureNumber}_{channel_name}_{plot_type} PSD",
                psd_save_path
            )

    except Exception as e:
        logger.error(f"Error in init_examination: {str(e)}")
        raise


def process_channel(data: np.ndarray, fs: int) -> np.ndarray:
    """Process single channel with filtering."""
    return butter_bandpass_filter(data, lowcut=1, highcut=127, fs=fs)


class SignalProcessor:
    """Class for signal processing operations."""

    @staticmethod
    def bipolar_reference(data: np.ndarray, gridmap: Dict) -> np.ndarray:
        """Apply bipolar referencing to data."""
        bipolar_data = np.zeros(data.shape)
        for i, e in enumerate(gridmap['Channel']):
            start, end = map(int, e.split(':'))
            for j in range(start, end):
                bipolar_data[:, j] = data[:, j] - data[:, j + 1]
            bipolar_data[:, end] = data[:, end]
        return bipolar_data

    @staticmethod
    def normalize_signal(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Normalize signal using RobustScaler."""
        normalized = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            scaler = RobustScaler()
            scaler.fit(reference[:, i].reshape(-1, 1))
            normalized[:, i] = scaler.transform(
                signal[:, i].reshape(-1, 1)
            ).squeeze()
        return normalized

    @staticmethod
    def apply_whitening(data: np.ndarray, lags: int = 1) -> np.ndarray:
        """Apply whitening using auto-regressive model."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError("Input data must be 1D or 2D array")

        whitened = np.zeros_like(data)
        for i in tqdm(range(data.shape[1]), desc="Whitening signals"):
            signal = data[:, i]
            model = AutoReg(signal, lags=lags).fit()
            predictions = model.predict(start=lags, end=len(signal) - 1)

            whitened[lags:, i] = signal[lags:] - predictions
            whitened[:lags, i] = signal[:lags] - np.mean(signal)

        return whitened.squeeze() if data.shape[1] == 1 else whitened

    @staticmethod
    def remove_line_noise(data: np.ndarray,
                          gridmap: Dict,
                          line_freqs: List[float] = [60, 100],
                          fs: int = 2000) -> np.ndarray:
        """Remove line noise from each electrode."""
        cleaned_data = np.array(data)
        for e in tqdm(gridmap['Channel'], desc="Removing line noise"):
            start, end = map(int, e.split(':'))
            section = cleaned_data[:, start:end + 1]
            for f0 in line_freqs:
                section, _ = dss.dss_line_iter(section, f0, fs)
            cleaned_data[:, start:end + 1] = section
        return cleaned_data


def preprocessing(dataset, data_folder: str):
    """Preprocess EEG dataset with comprehensive cleaning and normalization."""
    try:
        processor = SignalProcessor()

        # Load and scale data
        data_dict = {
            'ictal': dataset.ictal,
            'postictal': dataset.postictal,
            'interictal': dataset.interictal,
        }

        # Scale data
        for key in data_dict:
            data_dict[key] = data_dict[key] * 1e6

        # Apply bipolar referencing
        for key in data_dict:
            data_dict[key] = processor.bipolar_reference(
                data_dict[key], dataset.gridmap
            )

        # Remove line noise
        for key in data_dict:
            data_dict[key] = processor.remove_line_noise(
                data_dict[key], dataset.gridmap
            )

        # Apply bandpass filter
        for key in data_dict:
            for i in range(data_dict[key].shape[1]):
                data_dict[key][:, i] = butter_bandpass_filter(
                    data_dict[key][:, i], lowcut=1, highcut=255, fs=dataset.samplingRate
                )

        # Downsample
        try:
            factor = dataset.samplingRate // 512
            for key in data_dict:
                data_dict[key] = decimate(data_dict[key], factor, axis=0)
            dataset.downsample = True
            dataset.samplingRate = 512
        except Exception as e:
            logger.warning(f"Downsampling failed: {str(e)}")

        # Apply whitening and normalization
        try:
            for key in data_dict:
                data_dict[key] = processor.apply_whitening(data_dict[key])
            for key in data_dict:
                data_dict[key] = processor.normalize_signal(
                    data_dict[key], data_dict['interictal']
                )
        except Exception as e:
            logger.warning(f"Whitening/normalization failed: {str(e)}")

        # Save processed data
        for key in data_dict:
            setattr(dataset, key, data_dict[key])

        save_path = os.path.join(
            data_folder, f"seizure_{dataset.seizureNumber}_CLEANED.pkl"
        )
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Data saved to: {save_path}")

        return dataset

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


class EEGTimeSeriesFeatureExtractor:
    def __init__(self, sampling_rate=128, window_duration=0.05, window_step=0.025):
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_duration * sampling_rate)
        self.step_samples = int(window_step * sampling_rate)
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, min(200, sampling_rate // 2)),
        }

    def extract_half_wave_features(self, signal_window, amplitude_threshold=0.1):
        zero_crossings = np.where(np.diff(np.signbit(signal_window)))[0]
        if len(zero_crossings) <= 1:
            return {'hw_count': 0, 'hw_mean_amp': 0, 'hw_mean_duration': 0}
        half_wave_amps = []
        half_wave_durations = []
        for i in range(len(zero_crossings) - 1):
            start_idx = zero_crossings[i]
            end_idx = zero_crossings[i + 1]
            duration = end_idx - start_idx
            segment = signal_window[start_idx:end_idx]
            if len(segment) > 0:
                amplitude = np.max(np.abs(segment))
                if amplitude >= amplitude_threshold:
                    half_wave_amps.append(amplitude)
                    half_wave_durations.append(duration)
        return {
            'hw_count': len(half_wave_amps),
            'hw_mean_amp': np.mean(half_wave_amps) if half_wave_amps else 0,
            'hw_mean_duration': np.mean(half_wave_durations) / self.sampling_rate if half_wave_durations else 0
        }

    def extract_line_length(self, signal_window):
        return np.sum(np.abs(np.diff(signal_window)))

    def extract_area(self, signal_window):
        return np.sum(np.abs(signal_window))

    def _kurtosis(self, x):
        n = len(x)
        if n < 4:
            return 0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return 0
        m4 = np.sum((x - mean) ** 4) / n
        return m4 / (std ** 4) - 3

    def _skewness(self, x):
        n = len(x)
        if n < 3:
            return 0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return 0
        m3 = np.sum((x - mean) ** 3) / n
        return m3 / (std ** 3)

    def extract_features_from_window(self, signal_window):
        features = {}
        if len(signal_window) < 10 or np.all(signal_window == 0):
            return {key: 0 for key in [
                'mean', 'std', 'median', 'iqr', 'skew', 'kurtosis', 'range', 'rms',
                'zero_crossings', 'hw_count', 'hw_mean_amp', 'hw_mean_duration',
                'line_length', 'area', 'spectral_entropy', 'total_power'
            ] + [f'power_{b}' for b in self.freq_bands]}

        features['mean'] = np.mean(signal_window)
        features['std'] = np.std(signal_window)
        features['median'] = np.median(signal_window)
        features['iqr'] = np.percentile(signal_window, 75) - np.percentile(signal_window, 25)
        features['range'] = np.max(signal_window) - np.min(signal_window)
        features['rms'] = np.sqrt(np.mean(signal_window ** 2))
        features['zero_crossings'] = np.sum(np.diff(np.signbit(signal_window).astype(int)) != 0)

        features['skew'] = self._skewness(signal_window)
        features['kurtosis'] = self._kurtosis(signal_window)

        features.update(self.extract_half_wave_features(signal_window))
        features['line_length'] = self.extract_line_length(signal_window)
        features['area'] = self.extract_area(signal_window)

        try:
            windowed_signal = signal_window * np.hamming(len(signal_window))
            fft_vals = fft(windowed_signal)
            fft_abs = np.abs(fft_vals[:len(signal_window) // 2])
            fft_abs = fft_abs / len(signal_window)
            freq_bins = np.fft.fftfreq(len(signal_window), 1 / self.sampling_rate)[:len(signal_window) // 2]

            total_power = 0
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                band_mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)
                band_power = np.sum(fft_abs[band_mask] ** 2)
                features[f'power_{band_name}'] = band_power
                total_power += band_power

            features['total_power'] = total_power

            if total_power > 0:
                power_spectrum = fft_abs ** 2
                pxx_norm = power_spectrum / np.sum(power_spectrum)
                features['spectral_entropy'] = -np.sum(pxx_norm * np.log2(pxx_norm + 1e-10))
            else:
                features['spectral_entropy'] = 0

        except Exception:
            for band_name in self.freq_bands:
                features[f'power_{band_name}'] = 0
            features['spectral_entropy'] = 0
            features['total_power'] = 0

        return features

    def extract_time_series_features(self, signal_segment):
        if len(signal_segment) < self.window_samples:
            raise ValueError(f"Segment too short: {len(signal_segment)} < {self.window_samples}")
        window_features_list = []
        window_times = []
        for start_idx in range(0, len(signal_segment) - self.window_samples + 1, self.step_samples):
            end_idx = start_idx + self.window_samples
            window = signal_segment[start_idx:end_idx]
            window_time = start_idx / self.sampling_rate
            features = self.extract_features_from_window(window)
            window_features_list.append(features)
            window_times.append(window_time)
        feature_names = list(window_features_list[0].keys()) if window_features_list else []
        feature_array = np.zeros((len(window_features_list), len(feature_names)))
        for i, feat in enumerate(window_features_list):
            for j, name in enumerate(feature_names):
                feature_array[i, j] = feat[name]
        feature_array = RobustScaler().fit_transform(feature_array)
        return feature_array, feature_names, window_times



def extract_sEEG_features(seizure_object, sampling_rate=128, window_duration=0.05, window_step=0.025):
    """
    Extract features from seizure object and store them back in the same object.

    Args:
        seizure_object: Object containing ictal, postictal, and interictal data
                        Each field should be a 3D array with shape [segments, time, channels]
        sampling_rate: Sampling rate of the EEG data
        window_duration: Duration of sliding window in seconds (default: 50ms)
        window_step: Step size for window sliding in seconds (default: 25ms)

    Returns:
        The same seizure object with added transformed data fields
    """
    # Initialize feature extractor
    feature_extractor = EEGTimeSeriesFeatureExtractor(
        sampling_rate=sampling_rate,
        window_duration=window_duration,
        window_step=window_step
    )

    # Process different seizure states
    states = ['ictal', 'postictal', 'interictal']

    # Store feature names
    seizure_object.feature_names = None

    for state in states:
        # Get original data
        original_data = getattr(seizure_object, state)
        n_segments, segment_length, n_channels = original_data.shape

        print(f"Processing {n_segments} {state} segments × {n_channels} channels")

        # Store for transformed data
        # We'll create a 4D array: [segments, channels, windows, features]
        # First, we need to know the number of windows and features

        # Check the shape of feature array for one segment
        segment = original_data[0, :, 0]  # First segment, first channel
        feature_array, feature_names, _ = feature_extractor.extract_time_series_features(segment)

        if seizure_object.feature_names is None:
            seizure_object.feature_names = feature_names

        n_windows, n_features = feature_array.shape

        # Initialize the transformed data array
        transformed_data = np.zeros((n_segments, n_channels, n_windows, n_features))
        window_times_data = np.zeros((n_segments, n_channels, n_windows))

        # Process all segments and channels
        for segment_idx in range(n_segments):
            for channel_idx in range(n_channels):
                # Extract segment
                segment = original_data[segment_idx, :, channel_idx]

                # Extract features
                feature_array, _, window_times = feature_extractor.extract_time_series_features(segment)

                # Store in the transformed data array
                transformed_data[segment_idx, channel_idx] = feature_array
                window_times_data[segment_idx, channel_idx] = window_times

        # Store transformed data in the object
        setattr(seizure_object, f"{state}_transformed", transformed_data)
        setattr(seizure_object, f"{state}_window_times", window_times_data)
        setattr(seizure_object, f"{state}_feature_samplingRate", int(1/window_step))

    # Save seizure object
    save_path = os.path.join(f'data/P{seizure_object.patNo}', f"seizure_{seizure_object.seizureNumber}_combined.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(seizure_object, f)
        print(f"Saved transformed seizure data to {save_path}")

    return seizure_object


def setup_and_train_models(
        data_folder: str,
        model_folder: str,
        model_names: list = None,
        train: bool = False,
        input_type='raw',
        data_aug=False,
        params: dict = None,
        hyperparameter_search: bool = False,
        n_trials: int = 20,
        search_space: dict = None
):
    """
    Set up and train selected models for seizure detection with optional hyperparameter search

    Args:
        data_folder: Path to data folder
        model_folder: Path to save/load model checkpoints
        model_names: List of model names to use ['CNN1D', 'Wavenet', 'LSTM', 'EnhancedResNet']
                    If None, uses all available models
        train: Whether to train models or load pretrained weights
        input_type: Type of input data ('raw', 'transformed', or 'combined')
        data_aug: Whether to use data augmentation
        params: Dictionary of training parameters (optional)
        hyperparameter_search: Whether to perform hyperparameter search
        n_trials: Number of trials for hyperparameter search
        search_space: Dictionary defining the hyperparameter search space (optional)

    Returns:
        dict: Dictionary containing trained models and their performance metrics
    """

    # Available models
    AVAILABLE_MODELS = {
        'CNN1D': lambda ch, ts, lr: CNN1D(input_dim=ch, kernel_size=ts, output_dim=2, lr=lr),
        'Wavenet': lambda ch, ts, lr: Wavenet(input_dim=ch, output_dim=2, kernel_size=ts, lr=lr),
        'LSTM': lambda ch, ts, lr: LSTM(input_dim=ch, output_dim=2, lr=lr),
        'S4': lambda ch, ts, lr: S4Model(d_input=ch, d_output=2, lr=lr),
        'ResNet': lambda ch, ts, lr: ResNet(input_dim=ch, n_classes=2, lr=lr, kernel_size=ts, base_filters=32),
        'EnhancedResNet': lambda ch, ts, lr: EnhancedResNet(input_dim=ch, kernel_size=ts, lr=lr)
    }

    # Default parameters
    default_params = {
        'epochs': 40,
        'checkpoint_freq': 10,
        'lr': 0.001,
        'batch_size': 4096,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'patience': 7,
        'scheduler_patience': 5,
        'gradient_clip': 1.0
    }

    # Update defaults with provided parameters
    if params is not None:
        default_params.update(params)
    params = default_params

    # Use all models if none specified
    if model_names is None:
        model_names = list(AVAILABLE_MODELS.keys())
    else:
        # Validate model names
        invalid_models = [m for m in model_names if m not in AVAILABLE_MODELS]
        if invalid_models:
            raise ValueError(f"Invalid model names: {invalid_models}. "
                             f"Available models are: {list(AVAILABLE_MODELS.keys())}")

    # Create model folder if it doesn't exist
    os.makedirs(model_folder, exist_ok=True)

    def prepare_data(data_folder: str, batch_size: int, data_aug=data_aug, input_type=input_type):
        """Prepare datasets and dataloaders"""
        seizure_across_patients = load_seizure_across_patients(data_folder)
        ml_datasets = [create_dataset(seizure, batch_size=batch_size, input_type=input_type)
                       for seizure in seizure_across_patients]
        train_loader, val_loader = combine_loaders(ml_datasets, batch_size=batch_size)

        # Get the shape from a sample
        sample_batch = next(iter(train_loader))
        channels, time_steps = sample_batch['data'][0].shape

        return train_loader, val_loader, channels, time_steps

    def initialize_models(model_names: list, channels: int, time_steps: int, lr: float):
        """Initialize selected models"""
        return {
            name: AVAILABLE_MODELS[name](channels, time_steps, lr)
            for name in model_names
        }

    try:
        # Prepare data
        train_loader, val_loader, channels, time_steps = prepare_data(
            data_folder, params['batch_size']
        )

        # Dictionary to store results
        results = {
            'models': {},
            'training_history': {},
            'performance': {},
            'parameters': params,
            'best_hyperparameters': {}
        }

        # Initialize default search space if not provided
        if search_space is None and hyperparameter_search:
            search_space = {
                'lr': {'type': 'loguniform', 'range': [1e-5, 1e-2]},
                'batch_size': {'type': 'categorical', 'values': [512, 1024, 2048, 4096]},
                'gradient_clip': {'type': 'uniform', 'range': [0.1, 2.0]},
                'dropout': {'type': 'uniform', 'range': [0.1, 0.5]},
                'kernel_size': {'type': 'categorical', 'values': [32, 64, 128, 256, 512]},
            }

            # Add EnhancedResNet specific parameters if included in model_names
            if 'EnhancedResNet' in model_names:
                search_space.update({
                    'gamma': {'type': 'uniform', 'range': [0.1, 1.0]},  # Anatomical constraint weight
                    'delta': {'type': 'uniform', 'range': [0.1, 1.0]},  # Temporal consistency weight
                })

        # If hyperparameter search is enabled, find the best parameters for each model
        if hyperparameter_search and train:
            for model_name in model_names:
                print(f"\nPerforming hyperparameter search for {model_name}...")

                # Setup hyperparameter search for the current model
                best_params = hyperparameter_search_for_model(
                    model_name=model_name,
                    model_creator=AVAILABLE_MODELS[model_name],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    channels=channels,
                    time_steps=time_steps,
                    n_trials=n_trials,
                    search_space=search_space,
                    device=params['device'],
                    model_folder=model_folder
                )

                # Update the parameters for this model
                model_params = params.copy()
                model_params.update(best_params)
                results['best_hyperparameters'][model_name] = best_params

                # Initialize model with best parameters
                if model_name == 'EnhancedResNet':
                    model = EnhancedResNet(
                        input_dim=channels,
                        base_filters=32,
                        kernel_size=model_params.get('kernel_size', time_steps),
                        dropout=model_params.get('dropout', 0.2),
                        lr=model_params['lr'],
                        gamma=model_params.get('gamma', 0.5),  # Use optimized gamma if found
                        delta=model_params.get('delta', 0.5)  # Use optimized delta if found
                    )
                else:
                    model = AVAILABLE_MODELS[model_name](channels, time_steps, model_params['lr'])

                # Update dropout if needed
                if 'dropout' in best_params and hasattr(model, 'dropout'):
                    try:
                        for module in model.modules():
                            if isinstance(module, nn.Dropout):
                                module.p = best_params['dropout']
                    except:
                        logger.warning(f"Could not set dropout for {model_name}")

                # Train the model with the best parameters
                print(f"\nTraining {model_name} with best parameters: {best_params}")
                train_loss, val_loss, val_accuracy = train_using_optimizer(
                    model=model,
                    trainloader=train_loader,
                    valloader=val_loader,
                    save_location=model_folder,
                    epochs=params['epochs'],
                    device=params['device'],
                    patience=model_params['patience'],
                    scheduler_patience=model_params.get('scheduler_patience', 5),
                    gradient_clip=model_params['gradient_clip'],
                    checkpoint_freq=params['checkpoint_freq']
                )

                results['models'][model_name] = model
                results['training_history'][model_name] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }

        elif train:
            # Standard training without hyperparameter search
            # Initialize models
            models = initialize_models(model_names, channels, time_steps, params['lr'])
            results['models'] = models

            # Train each model
            for name, model in models.items():
                print(f"\nTraining {name}...")

                if isinstance(model, EnhancedResNet):
                    train_func = train_using_optimizer_with_masks
                else:
                    train_func = train_using_optimizer

                train_loss, val_loss, val_accuracy = train_func(
                    model=model,
                    trainloader=train_loader,
                    valloader=val_loader,
                    save_location=model_folder,
                    epochs=params['epochs'],
                    device=params['device'],
                    patience=params['patience'],
                    scheduler_patience=params['scheduler_patience'],
                    gradient_clip=params['gradient_clip'],
                    checkpoint_freq=params['checkpoint_freq']
                )

                results['training_history'][name] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }

        else:
            # Load pretrained models
            models = initialize_models(model_names, channels, time_steps, params['lr'])
            results['models'] = models

            # Load pretrained weights
            for name, model in models.items():
                checkpoint_path = os.path.join(model_folder, f"{name}_best.pth")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=params['device'])
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded checkpoint for {name}")
                else:
                    print(f"Warning: No checkpoint found for {name}")

        # Evaluate all models
        for name, model in results['models'].items():
            loss, accuracy = evaluate_model(model, val_loader, params['device'])
            results['performance'][name] = {
                'loss': loss,
                'accuracy': accuracy
            }

        # Plot training history if available
        if train and results['training_history']:
            # Plot training loss
            plt.figure(figsize=(10, 5))
            for name in results['training_history'].keys():
                plt.plot(results['training_history'][name]['train_loss'],
                         label=f"{name} Training Loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss vs Epoch")
            plt.savefig(os.path.join(model_folder, 'training_loss.png'))
            plt.close()

            # Plot validation accuracy
            plt.figure(figsize=(10, 5))
            for name in results['training_history'].keys():
                plt.plot(results['training_history'][name]['val_accuracy'],
                         label=f"{name} Validation Accuracy")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy vs Epoch")
            plt.savefig(os.path.join(model_folder, 'validation_accuracy.png'))
            plt.close()

        return results, results['models']

    except Exception as e:
        logger.error(f"Error in setup and training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def analyze_seizure_propagation(
        patient_no: int,
        seizure_no: int,
        model,
        data_folder: str = 'data',
        marking_file: str = 'data/Seizure_Onset_Type_ML_USC.xlsx',
        params: Optional[Dict] = None,
        save_results_ind: bool = True,
        recalculate_features: bool = False
) -> Dict:
    """
    Analyze seizure propagation for a specific patient and seizure

    Args:
        patient_no: Patient number
        seizure_no: Seizure number
        model: Neural network model for seizure detection
        data_folder: Base path to data folder
        marking_file: Path to seizure marking Excel file
        params: Dictionary of parameters for analysis (optional)

    Returns:
        Dict containing analysis results and performance metrics
    """
    # Default parameters
    default_params = {
        'threshold': 0.8,
        'smooth_window': 50,
        'n_seconds': 80,
        'seizure_start': 60,
        'seizure_plot_time': 10, # How many seconds to plot before seizrue starts
        'overlap': 0.8,
        'device': 'cuda:0'
    }

    # Update defaults with provided parameters
    if params is not None:
        default_params.update(params)
    params = default_params

    prob_fs = int(1/(1 - params['overlap']))

    # Set up paths
    single_seizure_folder = os.path.join(data_folder, f"P{patient_no}")
    save_folder = os.path.join("result", f"P{patient_no}", f"Seizure{seizure_no}")
    os.makedirs(save_folder, exist_ok=True)

    # Get model name
    model_name = model.__class__.__name__

    def load_matter_file(folder: str) -> Dict:
        """Load Matter file for seizure"""
        matter_file = os.path.join(folder, "matter.csv")
        matter = pd.read_csv(matter_file)
        return matter

    def load_seizure_data() -> Tuple[object, List[str], List[str]]:
        """Load seizure data and channel information"""
        # Load seizure marking data
        seizure_marking = pd.read_excel(marking_file)

        # Find seizure-related channels
        seizure_channels, seizure_onset_channels = find_seizure_related_channels(
            seizure_marking, seizure_no, patient_no
        )

        # Load seizure data
        seizure_obj = load_single_seizure(single_seizure_folder, seizure_no)

        if not hasattr(seizure_obj, 'ictal_transformed') or recalculate_features:
            seizure_obj = extract_sEEG_features(seizure_obj, sampling_rate=seizure_obj.samplingRate)

        # Load Matter file
        if not hasattr(seizure_obj, 'matter'):
            seizure_obj.matter = load_matter_file(single_seizure_folder)

        return seizure_obj, seizure_channels, seizure_onset_channels

    def extract_grey_matter_channels(matter: pd.DataFrame, seizure_channels: List[str]) -> List[str]:
        """Extract grey and 'A' matter channels from Matter file"""
        # Get channels that are either grey matter ('G') or 'A' type
        selected_matter = matter[matter['MatterType'].isin(['G', 'A'])]
        selected_matter_channel_set = set(selected_matter['ElectrodeName'].values)

        # Preserve original order from seizure_channels by filtering
        selected_channels = [channel for channel in seizure_channels if channel in selected_matter_channel_set]

        return selected_channels

    def process_data(seizure_obj) -> Tuple[np.ndarray, np.ndarray, float]:
        """Process raw seizure data"""
        fs = seizure_obj.samplingRate
        if not hasattr(seizure_obj, 'ictal_transformed'):
            ictal_data = seizure_obj.ictal_
            preictal_data = seizure_obj.preictal2

            # Reshape and combine data
            ictal_combined = ictal_data.reshape(-1, ictal_data.shape[2])
            total_data = np.concatenate((preictal_data, ictal_combined), axis=0)

        else:
            ictal_data = seizure_obj.ictal_transformed
            preictal_data = seizure_obj.interictal_transformed

            # Reshape and combine data
            ictal_combined = ictal_data.transpose(0, 2, 1, 3).reshape(ictal_data.shape[0] * ictal_data.shape[2],
                                                                      ictal_data.shape[1], ictal_data.shape[3])
            preictal_data = preictal_data.transpose(0, 2, 1, 3).reshape(preictal_data.shape[0] * preictal_data.shape[2],
                                                                        preictal_data.shape[1], preictal_data.shape[3])

            total_data = np.concatenate((preictal_data, ictal_combined))

        # Split data into windows
        total_windows = split_data(total_data, 40, overlap=params['overlap'])

        return total_data, total_windows, fs

    def compute_probabilities(data: np.ndarray, model, device: str) -> np.ndarray:
        """
        Compute seizure probabilities for each channel.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data with shape (chunks, fs, channel) or (chunks, fs, channel, features)
        model : torch model
            The seizure detection model
        device : str
            The device to run the model on ('cpu' or 'cuda')

        Returns:
        --------
        numpy.ndarray
            Probability matrix with shape (chunks, channel)
        """
        # Determine if the input is 3D or 4D
        is_4d = len(data.shape) == 4

        # Get dimensions
        chunks = data.shape[0]
        fs = data.shape[1]
        n_channels = data.shape[2]

        # Initialize probability matrix
        prob_matrix = np.zeros((chunks, n_channels))

        for channel in range(n_channels):
            if is_4d:
                # 4D data: [chunks, fs, channel, features]
                channel_data = data[:, :, channel, :]
                input_data = np.transpose(channel_data, (0, 2, 1))
            else:
                # 3D data: [chunks, fs, channel]
                # Extract and reshape to [chunks, 1, fs]
                channel_data = data[:, :, channel]
                input_data = channel_data.reshape(chunks, 1, fs)  # [chunks, 1, fs]

            # Convert to tensor and move to device
            input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

            # Compute probabilities
            prob_matrix[:, channel] = output_to_probability(model, input_data, device)

        return prob_matrix

    def smooth_and_rank_channels(prob_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth probability matrix and rank channels"""
        # Smooth probabilities
        prob_smoothed = np.zeros_like(prob_matrix)
        for i in range(prob_matrix.shape[1]):
            prob_smoothed[:, i] = np.convolve(
                prob_matrix[:, i],
                np.ones(params['smooth_window']) / params['smooth_window'],
                mode='same'
            )

        # Get section after seizure start
        prob_section = prob_smoothed[params['seizure_start'] * prob_fs:, :]

        # Find first threshold crossing
        first_threshold = np.argmax(prob_section > params['threshold'], axis=0)
        first_threshold[first_threshold == 0] = len(prob_section)

        # Sort channels
        sorted_indices = np.argsort(first_threshold)[::-1]

        return prob_smoothed, sorted_indices

    def evaluate_performance(channel_names: List[str],
                             seizure_channels: List[str],
                             seizure_onset_channels: List[str]) -> Dict:
        """Evaluate channel detection performance"""
        n_seizure = len(seizure_channels)
        n_onset = len(seizure_onset_channels)

        correct_channels = sum(ch in seizure_channels
                               for ch in channel_names[:n_seizure]) / n_seizure
        correct_onset = sum(ch in seizure_onset_channels
                            for ch in channel_names[:n_onset]) / n_onset

        return {
            'accuracy_all_channels': correct_channels,
            'accuracy_onset_channels': correct_onset,
            'seizure_channels': seizure_channels,
            'onset_channels': seizure_onset_channels,
            'detected_channels': channel_names[:n_seizure],
            'detected_onset': channel_names[:n_onset]
        }

    # Main analysis pipeline
    try:
        # Load data
        seizure_obj, seizure_channels, seizure_onset_channels = load_seizure_data()

        # Process data
        total_data, windowed_data, fs = process_data(seizure_obj)

        # Compute probabilities
        probabilities = compute_probabilities(windowed_data, model, params['device'])

        # Smooth and rank channels
        prob_smoothed, sorted_indices = smooth_and_rank_channels(probabilities)

        # Only start from seizure_plot_time seconds before the actual seizure
        prob_smoothed = prob_smoothed[(params['seizure_start'] - params['seizure_plot_time']) * prob_fs:, :]

        # Map channel numbers to names
        channel_names = map_seizure_channels(sorted_indices[::-1],
                                             seizure_obj.gridmap)

        grey_channel_names = extract_grey_matter_channels(seizure_obj.matter, channel_names)
        grey_seizure_channels = extract_grey_matter_channels(seizure_obj.matter, seizure_channels)
        grey_onset_channels = extract_grey_matter_channels(seizure_obj.matter, seizure_onset_channels)

        # Evaluate performance
        performance = evaluate_performance(grey_channel_names,
                                           grey_seizure_channels,
                                           grey_onset_channels)

        # Create time axes for plotting
        time_prob = np.arange(prob_smoothed.shape[0]) * (1 - params['overlap'])

        # Save plots
        plot_time_limited_heatmap(
            data=prob_smoothed[:, sorted_indices].T,
            time_axis=time_prob,
            n_seconds=params['n_seconds'],
            preictal_boundary=params['seizure_plot_time'] * prob_fs,
            title=f"{model_name} Probability (Reranked)",
            cmap='hot',
            save_path=os.path.join(save_folder, f"{model_name}ProbabilityReranked.png"),
            flip_yaxis=False
        )

        plot_time_limited_heatmap(
            data=prob_smoothed.T,
            time_axis=time_prob,
            n_seconds=params['n_seconds'],
            preictal_boundary=params['seizure_plot_time'] * prob_fs,
            title=f"{model_name} Probability",
            cmap='hot',
            save_path=os.path.join(save_folder, f"{model_name}ProbabilityNoranked.png"),
            flip_yaxis=True
        )

        raw_data_ictal = seizure_obj.ictal
        raw_data_preicatal = seizure_obj.interictal

        # Reshape the data
        raw_data_ictal = raw_data_ictal.reshape(-1, raw_data_ictal.shape[2])[:(params['n_seconds'] - params['seizure_plot_time']) * seizure_obj.samplingRate]
        raw_data_preicatal = raw_data_preicatal.reshape(-1, raw_data_preicatal.shape[2])[-params['seizure_plot_time'] * seizure_obj.samplingRate:]

        # Combine the data
        sub_data = np.concatenate((raw_data_preicatal, raw_data_ictal), axis=0)

        # Save raw EEG plot
        # sub_data = total_data[:int(fs * params['n_seconds']), :]
        fig = plot_eeg_style(sub_data.T, fs, spacing_factor=2, color='black', linewidth=0.5)
        fig.savefig(os.path.join(save_folder, "Raw_EEG.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save probability plot
        sub_prob = probabilities[:int(5 * params['n_seconds']), :]
        fig = plot_eeg_style(sub_prob.T, 5, spacing_factor=2, color='black', linewidth=0.5)
        fig.savefig(os.path.join(save_folder, f"{model_name}_Probability.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Return results
        results = {
            'performance': performance,
            'probabilities': probabilities,
            'smoothed_probabilities': prob_smoothed,
            'sorted_indices': sorted_indices[::-1],
            'channel_names': channel_names,
            'true_seizure_channels': map_channels_to_numbers(sorted_indices, channel_names, seizure_channels),
            'true_onset_channels': map_channels_to_numbers(sorted_indices, channel_names, seizure_onset_channels),
            'parameters': params,
            'sampling_rate': fs,
            'save_folder': save_folder
        }

        # Save results if requested
        if save_results_ind:
            results['results_folder'] = save_results(results, save_folder, model_name)
            print(f"Results saved to: {results['results_folder']}")

        return results

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise


def save_results(results: dict, save_folder: str, model_name: str = None) -> str:
    """
    Save analysis results to files

    Args:
        results: Dictionary containing analysis results
        save_folder: Folder to save results
        model_name: Name of the model used (optional)

    Returns:
        str: Path to the saved results
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_name or results.get('model_name', 'unknown_model')

    # Create results folder if it doesn't exist
    results_folder = os.path.join(save_folder, 'analysis_results')
    os.makedirs(results_folder, exist_ok=True)

    # Prepare data for saving
    save_data = {
        'performance': results['performance'],
        'parameters': results['parameters'],
        'channel_names': results['channel_names'],
        'timestamp': timestamp,
        'model_name': model_name
    }

    # Save readable metrics to JSON
    json_path = os.path.join(results_folder, f'metrics_{model_name}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'performance': results['performance'],
            'parameters': results['parameters'],
            'channel_names': list(results['channel_names']),
            'timestamp': timestamp,
            'model_name': model_name
        }, f, indent=4)

    # Save full results (including numpy arrays) to pickle
    pickle_path = os.path.join(results_folder, f'full_results_{model_name}_{timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Save a summary text file
    summary_path = os.path.join(results_folder, f'summary_{model_name}_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Analysis Summary - {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Channel detection accuracy: {results['performance']['accuracy_all_channels']:.4f}\n")
        f.write(f"Onset channel detection accuracy: {results['performance']['accuracy_onset_channels']:.4f}\n\n")
        f.write("Detected Channels (in order):\n")
        for i, channel in enumerate(results['performance']['detected_channels'], 1):
            f.write(f"{i}. {channel}\n")
        f.write("\nActual Seizure Channels:\n")
        for channel in results['performance']['seizure_channels']:
            f.write(f"- {channel}\n")
        f.write("\nParameters Used:\n")
        for param, value in results['parameters'].items():
            f.write(f"{param}: {value}\n")

    return results_folder


def load_results(results_path: str) -> dict:
    """
    Load saved analysis results

    Args:
        results_path: Path to the saved results file (.pkl)

    Returns:
        dict: Loaded results dictionary
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results