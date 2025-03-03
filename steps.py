import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, decimate
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import signal
from scipy.fftpack import fft
from statsmodels.tsa.ar_model import AutoReg
# from meegkit import dss
from tqdm import tqdm
import pickle
from typing import Tuple, Optional, List, Dict
import pandas as pd
import logging
import torch
import json
import datetime

from utils import (butter_bandpass_filter,split_data, map_seizure_channels,
                   find_seizure_related_channels, map_channels_to_numbers)
from plotFun import plot_time_limited_heatmap, plot_eeg_style
from datasetConstruct import (combine_loaders,
                              load_seizure_across_patients, create_dataset,
                              EDFData, load_single_seizure)
from models import (train_using_optimizer, evaluate_model, output_to_probability, Wavenet, CNN1D,
                    LSTM, S4Model, WaveResNet)

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
            # for f0 in line_freqs:
            #     section, _ = dss.dss_line_iter(section, f0, fs)
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
            'preictal2': dataset.preictal2,
            'postictal2': dataset.postictal2,
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
                    data_dict[key][:, i], lowcut=1, highcut=127, fs=dataset.samplingRate
                )

        # Downsample
        try:
            factor = dataset.samplingRate // 128
            for key in data_dict:
                data_dict[key] = decimate(data_dict[key], factor, axis=0)
            dataset.downsample = True
            dataset.samplingRate = 128
        except Exception as e:
            logger.warning(f"Downsampling failed: {str(e)}")

        # Apply whitening and normalization
        try:
            for key in data_dict:
                data_dict[key] = processor.apply_whitening(data_dict[key])
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
    """
    Extracts time-series features from EEG signals by calculating features
    in sliding windows across the signal.

    Features include:
    - Half-wave features
    - Line length
    - Signal area (energy)
    - Frequency band powers (delta, theta, alpha, beta, gamma)
    """

    def __init__(self, sampling_rate=256, window_duration=0.05, window_step=0.025):
        """
        Initialize the feature extractor with moving window parameters.

        Args:
            sampling_rate: Number of samples per second in the EEG data
            window_duration: Duration of sliding window in seconds (default: 50ms)
            window_step: Step size for window sliding in seconds (default: 25ms)
        """
        self.sampling_rate = sampling_rate

        # Calculate window sizes in samples
        self.window_samples = int(window_duration * sampling_rate)
        self.step_samples = int(window_step * sampling_rate)

        # Define frequency bands (Hz)
        self.freq_bands = {
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200),
        }

    def extract_half_wave_features(self, signal_window):
        """
        Extract features based on half-wave analysis.

        Args:
            signal_window: 1D array of signal values

        Returns:
            Dictionary of half-wave features
        """
        # Apply bandpass filter (1-30Hz is typical for seizure detection)
        sos = signal.butter(4, [1, 30], 'bandpass', fs=self.sampling_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, signal_window)

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(filtered_signal)))[0]

        # If no zero crossings found, return zeros
        if len(zero_crossings) <= 1:
            return {
                'hw_count': 0,
                'hw_mean_amp': 0,
                'hw_max_amp': 0,
                'hw_mean_duration': 0
            }

        # Compute half-wave amplitudes and durations
        half_wave_amps = []
        half_wave_durations = []

        for i in range(len(zero_crossings) - 1):
            start_idx = zero_crossings[i]
            end_idx = zero_crossings[i + 1]

            # Calculate half-wave duration in samples
            duration = end_idx - start_idx
            half_wave_durations.append(duration)

            # Get max amplitude in this half-wave
            segment = filtered_signal[start_idx:end_idx]
            if len(segment) > 0:
                half_wave_amps.append(np.max(np.abs(segment)))

        # Return features
        return {
            'hw_count': len(half_wave_amps),
            'hw_mean_amp': np.mean(half_wave_amps) if half_wave_amps else 0,
            'hw_max_amp': np.max(half_wave_amps) if half_wave_amps else 0,
            'hw_mean_duration': np.mean(half_wave_durations) / self.sampling_rate if half_wave_durations else 0
        }

    def extract_line_length(self, signal_window):
        """
        Compute line length feature.

        Args:
            signal_window: 1D array of signal values

        Returns:
            Line length value
        """
        return np.sum(np.abs(np.diff(signal_window)))

    def extract_area(self, signal_window):
        """
        Compute area under the curve (signal energy).

        Args:
            signal_window: 1D array of signal values

        Returns:
            Area value
        """
        return np.sum(np.abs(signal_window))

    def extract_frequency_bands(self, signal_window):
        """
        Extract power in standard frequency bands using FFT.

        Args:
            signal_window: 1D array of signal values

        Returns:
            Dictionary of band powers
        """
        # Apply Hamming window to reduce spectral leakage
        windowed_signal = signal_window * np.hamming(len(signal_window))

        # Compute FFT
        fft_vals = fft(windowed_signal)
        fft_abs = np.abs(fft_vals[:len(signal_window) // 2])

        # Normalize by window length
        fft_abs = fft_abs / len(signal_window)

        # Calculate frequency bins
        freq_bins = np.fft.fftfreq(len(signal_window), 1 / self.sampling_rate)[:len(signal_window) // 2]

        # Calculate band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            band_mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)
            band_power = np.sum(fft_abs[band_mask] ** 2)
            band_powers[f'power_{band_name}'] = band_power

        # Calculate relative band powers
        total_power = sum(band_powers.values())
        if total_power > 0:
            for band_name in self.freq_bands.keys():
                band_powers[f'rel_power_{band_name}'] = band_powers[f'power_{band_name}'] / total_power
        else:
            for band_name in self.freq_bands.keys():
                band_powers[f'rel_power_{band_name}'] = 0

        return band_powers

    def extract_statistical_features(self, signal_window):
        """
        Extract basic statistical features from the signal.

        Args:
            signal_window: 1D array of signal values

        Returns:
            Dictionary of statistical features
        """
        return {
            'mean': np.mean(signal_window),
            'std': np.std(signal_window),
            'kurtosis': self._kurtosis(signal_window),
            'skewness': self._skewness(signal_window),
            'max': np.max(signal_window),
            'min': np.min(signal_window),
            'peak_to_peak': np.max(signal_window) - np.min(signal_window),
            'energy': np.sum(signal_window ** 2),
            'rms': np.sqrt(np.mean(signal_window ** 2))
        }

    def _kurtosis(self, x):
        """Compute kurtosis of a signal"""
        n = len(x)
        if n < 4:
            return 0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return 0
        m4 = np.sum((x - mean) ** 4) / n
        return m4 / (std ** 4) - 3  # -3 to make normal distribution have kurtosis=0

    def _skewness(self, x):
        """Compute skewness of a signal"""
        n = len(x)
        if n < 3:
            return 0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return 0
        m3 = np.sum((x - mean) ** 3) / n
        return m3 / (std ** 3)

    def extract_hjorth_parameters(self, signal_window):
        """
        Extract Hjorth parameters (activity, mobility, complexity).

        Args:
            signal_window: 1D array of signal values

        Returns:
            Dictionary of Hjorth parameters
        """
        # Calculate first and second derivatives
        diff1 = np.diff(signal_window)
        diff2 = np.diff(diff1)

        # Pad the derivatives to match original length
        diff1 = np.append(diff1, diff1[-1])
        diff2 = np.append(diff2, [diff2[-1], diff2[-1]])

        # Calculate variance of signal and derivatives
        var0 = np.var(signal_window)
        var1 = np.var(diff1)
        var2 = np.var(diff2)

        # Calculate Hjorth parameters
        activity = var0
        mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 and var1 > 0 else 0

        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }

    def extract_features_from_window(self, signal_window):
        """
        Extract all features from a single window of EEG data.

        Args:
            signal_window: 1D array of signal values

        Returns:
            Dictionary of all features
        """
        features = {}

        # Extract half-wave features
        features.update(self.extract_half_wave_features(signal_window))

        # Add line length feature
        features['line_length'] = self.extract_line_length(signal_window)

        # Add area feature
        features['area'] = self.extract_area(signal_window)

        # Add frequency band powers
        features.update(self.extract_frequency_bands(signal_window))

        return features

    def extract_time_series_features(self, signal_segment):
        """
        Extract time-series features from a 1-second segment using a moving window approach.
        Each window produces a feature vector, maintaining the time-series nature of the data.

        Args:
            signal_segment: 1D array of signal values (1-second segment)

        Returns:
            2D array of features with shape (n_windows, n_features)
            List of feature names
        """
        # Check if segment is long enough for sliding window
        if len(signal_segment) < self.window_samples:
            raise ValueError(
                f"Signal segment length ({len(signal_segment)}) is shorter than window size ({self.window_samples})")

        # Initialize list to store features from each window
        window_features_list = []
        window_times = []

        # Slide window through the segment
        for start_idx in range(0, len(signal_segment) - self.window_samples + 1, self.step_samples):
            end_idx = start_idx + self.window_samples
            window_time = start_idx / self.sampling_rate  # Time in seconds from start of segment

            # Extract window
            window = signal_segment[start_idx:end_idx]

            # Extract features from this window
            window_features = self.extract_features_from_window(window)

            # Add to list
            window_features_list.append(window_features)
            window_times.append(window_time)

        # Get feature names from the first window (if available)
        feature_names = list(window_features_list[0].keys()) if window_features_list else []

        # Convert list of dictionaries to 2D array
        # Each row is a window, each column is a feature
        feature_array = np.zeros((len(window_features_list), len(feature_names)))

        for i, features in enumerate(window_features_list):
            for j, feature_name in enumerate(feature_names):
                feature_array[i, j] = features[feature_name]

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

    # Save seizure object
    save_path = f"data/P{seizure_object.patNo}/seizure_combined.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(seizure_object, f)

    return seizure_object


def setup_and_train_models(
        data_folder: str,
        model_folder: str,
        model_names: list = None,
        train: bool = False,
        input_type='raw',
        data_aug = False,
        params: dict = None
):
    """
    Set up and train selected models for seizure detection

    Args:
        data_folder: Path to data folder
        model_folder: Path to save/load model checkpoints
        model_names: List of model names to use ['CNN1D', 'Wavenet', 'LSTM']
                    If None, uses all available models
        train: Whether to train models or load pretrained weights
        params: Dictionary of training parameters (optional)

    Returns:
        dict: Dictionary containing trained models and their performance metrics
    """
    # Available models
    AVAILABLE_MODELS = {
        'CNN1D': lambda ch, ts, lr: CNN1D(input_dim=ch, kernel_size=ts, output_dim=2, lr=lr),
        'Wavenet': lambda ch, ts, lr: Wavenet(input_dim=ch, output_dim=2, kernel_size=ts, lr=lr),
        'LSTM': lambda ch, ts, lr: LSTM(input_dim=ch, output_dim=2, lr=lr),
        'S4': lambda ch, ts, lr: S4Model(d_input=ch, d_output=2, lr=lr),
        'ResNet': lambda ch, ts, lr: WaveResNet(input_dim=ch, n_classes=2, lr=lr, kernel_size=ts)
    }

    # Default parameters
    default_params = {
        'epochs': 40,
        'checkpoint_freq': 10,
        'lr': 0.001,
        'batch_size': 4096,
        'device': 'cuda:0',
        'patience': 7,
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
        channels, time_steps = train_loader.dataset[0][0].shape
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

        # Initialize models
        models = initialize_models(model_names, channels, time_steps, params['lr'])

        # Dictionary to store results
        results = {
            'models': models,
            'training_history': {},
            'performance': {},
            'parameters': params
        }

        if train:
            # Train each model
            for name, model in models.items():
                print(f"\nTraining {name}...")

                train_loss, val_loss, val_accuracy = train_using_optimizer(
                    model=model,
                    trainloader=train_loader,
                    valloader=val_loader,
                    save_location=model_folder,
                    epochs=params['epochs'],
                    device=params['device'],
                    patience=params['patience'],
                    gradient_clip=params['gradient_clip'],
                    checkpoint_freq=params['checkpoint_freq']
                )

                results['training_history'][name] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }

        else:
            # Load pretrained weights
            for name, model in models.items():
                checkpoint_path = os.path.join(model_folder, f"{name}_best.pth")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(
                        torch.load(checkpoint_path)['model_state_dict']
                    )
                else:
                    print(f"Warning: No checkpoint found for {name}")

        # Evaluate all models
        for name, model in models.items():
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
            plt.savefig(os.path.join(model_folder, f'{name}training_loss.png'))
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
            plt.savefig(os.path.join(model_folder, f'{name}validation_accuracy.png'))
            plt.close()

        return results, models

    except Exception as e:
        print(f"Error in setup and training: {str(e)}")
        raise


def analyze_seizure_propagation(
        patient_no: int,
        seizure_no: int,
        model,
        data_folder: str = 'data',
        marking_file: str = 'data/Seizure_Onset_Type_ML_USC.xlsx',
        params: Optional[Dict] = None,
        save_results_ind: bool = True
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
        'seizure_start': 10,
        'overlap': 0.8,
        'device': 'cuda:0'
    }

    # Update defaults with provided parameters
    if params is not None:
        default_params.update(params)
    params = default_params

    # Set up paths
    single_seizure_folder = os.path.join(data_folder, f"P{patient_no}")
    save_folder = os.path.join("result", f"P{patient_no}", f"Seizure{seizure_no}")
    os.makedirs(save_folder, exist_ok=True)

    # Get model name
    model_name = model.__class__.__name__

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

        return seizure_obj, seizure_channels, seizure_onset_channels

    def process_data(seizure_obj) -> Tuple[np.ndarray, np.ndarray, float]:
        """Process raw seizure data"""
        fs = seizure_obj.samplingRate
        ictal_data = seizure_obj.ictal
        preictal_data = seizure_obj.preictal2

        # Reshape and combine data
        ictal_combined = ictal_data.reshape(-1, ictal_data.shape[2])
        total_data = np.concatenate((preictal_data, ictal_combined), axis=0)

        # Split data into windows
        total_windows = split_data(total_data, fs, overlap=params['overlap'])

        return total_data, total_windows, fs

    def compute_probabilities(data: np.ndarray, model, device: str) -> np.ndarray:
        """Compute seizure probabilities for each channel"""
        prob_matrix = np.zeros((data.shape[0], data.shape[2]))

        for channel in range(data.shape[2]):
            input_data = data[:, :, channel].reshape(-1, 1, data.shape[1])
            input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
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
        prob_section = prob_smoothed[params['seizure_start'] * 5:, :]

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

        # Map channel numbers to names
        channel_names = map_seizure_channels(sorted_indices[::-1],
                                             seizure_obj.gridmap)

        # Evaluate performance
        performance = evaluate_performance(channel_names,
                                           seizure_channels,
                                           seizure_onset_channels)

        # Create time axes for plotting
        time_prob = np.arange(probabilities.shape[0]) * 0.2

        # Save plots
        plot_time_limited_heatmap(
            data=prob_smoothed[:, sorted_indices].T,
            time_axis=time_prob,
            n_seconds=params['n_seconds'],
            preictal_boundary=50,
            title=f"{model_name} Probability (Reranked)",
            cmap='hot',
            save_path=os.path.join(save_folder, f"{model_name}ProbabilityReranked.png"),
            flip_yaxis=False
        )

        # Save raw EEG plot
        sub_data = total_data[:int(fs * params['n_seconds']), :]
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