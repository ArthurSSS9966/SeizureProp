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
import logging

from utils import butter_bandpass_filter

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