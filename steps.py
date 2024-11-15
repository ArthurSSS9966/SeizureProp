import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, decimate
from sklearn.preprocessing import RobustScaler
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

from utils import butter_bandpass_filter,split_data, map_seizure_channels, find_seizure_related_channels
from plotFun import plot_time_limited_heatmap, plot_eeg_style
from datasetConstruct import (combine_loaders,
                              load_seizure_across_patients, create_dataset,
                              EDFData, load_single_seizure)
from models import train_using_optimizer, evaluate_model, output_to_probability, Wavenet, CNN1D, LSTM

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
            'preictal2': dataset.preictal2,
            'postictal2': dataset.postictal2,
            'interictal': dataset.interictal,
        }

        # Scale data
        for key in data_dict:
            data_dict[key] = data_dict[key] * 1e6

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


def setup_and_train_models(
        data_folder: str,
        model_folder: str,
        model_names: list = None,
        train: bool = False,
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
        'LSTM': lambda ch, ts, lr: LSTM(input_dim=ch, output_dim=2, lr=lr)
    }

    # Default parameters
    default_params = {
        'epochs': 40,
        'checkpoint_freq': 5,
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

    def prepare_data(data_folder: str, batch_size: int):
        """Prepare datasets and dataloaders"""
        seizure_across_patients = load_seizure_across_patients(data_folder)
        ml_datasets = [create_dataset(seizure, batch_size=batch_size)
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
            'sorted_indices': sorted_indices,
            'channel_names': channel_names,
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