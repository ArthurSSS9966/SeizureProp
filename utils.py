from scipy.signal import butter, filtfilt, sosfilt
from scipy.signal import iirnotch
import pandas as pd
import re
import numpy as np
import os
import pickle


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''

    :param lowcut: low cut frequency
    :param highcut: high cut frequency
    :param fs: sampling frequency
    :param order: filter order
    :return:
    '''
    return butter(order, [lowcut, highcut], fs=fs, btype='band', analog=False, output='sos')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''

    :param data: 1D data vector
    :param lowcut: low cut frequency
    :param highcut: high cut frequency
    :param fs: sampling frequency
    :param order: filter order
    :return:
    '''
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def notch_filter(data, fs, freq, Q=30):
    '''
    Function to apply notch filter to the data
    :param data: 2D matrix with shape (n_samples, n_channels)
    :param Fs: Sampling frequency
    :param freqs: List of frequencies to notch out
    :param Q: Quality factor
    :return:
    '''
    b, a = iirnotch(freq, Q, fs)
    data = filtfilt(b, a, data, axis=0)

    return data


def process_edf_channels_and_create_new_gridmap(original_gridmap, ch_number, raw_data):
    # Step 1: Filter and process the channel list
    useful_channels = []
    for i, channel in enumerate(ch_number):
        if re.match(r'^[A-Za-z]+[0-9]+$', channel) and not channel.lower().startswith('c') and not channel.lower().startswith('dc'):
            useful_channels.append((channel, i))

    # Step 2: Create a dictionary to group channels by their label
    channel_groups = {}
    for channel, index in useful_channels:
        label = re.match(r'^[A-Za-z]+', channel).group()
        if label not in channel_groups:
            channel_groups[label] = []
        channel_groups[label].append((channel, index))

    # Step 3: Create new gridmap with the same format as the original and continuous channel numbering
    new_gridmap_data = []
    grid_id = 0
    current_channel = 0
    for label, channels in channel_groups.items():
        num_channels = len(channels)
        start_channel = current_channel
        end_channel = current_channel + num_channels - 1

        # Find the corresponding row in the original gridmap
        original_row = original_gridmap[original_gridmap['Label'] == label]
        if not original_row.empty:
            template = original_row['Template'].values[0]
            location = original_row['Location'].values[0]
            hemisphere = original_row['Hemisphere'].values[0]
        else:
            template = "Unknown"
            location = "Unknown"
            hemisphere = "Unknown"

        new_gridmap_data.append({
            'GridId': grid_id,
            'Template': template,
            'Location': location,
            'Hemisphere': hemisphere,
            'Label': label,
            'Channel': f"{start_channel}:{end_channel}",
            'GridElectrode': f"1:{num_channels}"
        })
        grid_id += 1
        current_channel = end_channel + 1

    new_gridmap = pd.DataFrame(new_gridmap_data)

    # Step 4: Update raw data
    useful_indices = [index for _, index in useful_channels]
    new_raw_data = raw_data[:, useful_indices]

    # Step 5: Create a mapping of channel names to their new indices
    channel_mapping = {channel: i for i, (channel, _) in enumerate(useful_channels)}

    return new_gridmap, new_raw_data


def find_seizure_annotations(raw):
    """
    Find the indices of seizure onset and offset annotations in raw EEG data.

    :param raw: Raw EEG data object with annotations
    :return: Tuple of (SZON_ind, SZOFF_ind)
    """
    # Convert annotations to a numpy array for easier processing
    descriptions = np.array(raw.annotations.description)

    # Find indices for seizure onset
    SZON_indices = np.where((descriptions == 'SZON') | (descriptions == 'SZ') | (descriptions == 'sz'))[0]

    # Find indices for seizure offset
    SZOFF_indices = np.where((descriptions == 'SZOFF') | (descriptions == 'end'))[0]

    # Check if we found both onset and offset
    if len(SZOFF_indices) == 0:
        SZOFF_indices = [len(descriptions) - 1]

    # Return the first occurrence of each
    return SZON_indices[0], SZOFF_indices[0]


def split_data(data, fs, overlap=0):
    '''
    Function to split 2D data into chunks of size fs with specified overlap

    Parameters:
    -----------
    data : numpy.ndarray
        Input 2D data to be split, shape (samples, features)
    fs : int
        Window size (number of samples per chunk)
    overlap : float
        Overlap between consecutive windows (0 to 1), default is 0
        e.g., 0.5 means 50% overlap

    Returns:
    --------
    numpy.ndarray
        Array of overlapped chunks with shape (n_chunks, fs, n_features)
    '''
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")

    # Calculate step size based on overlap
    step = int(fs * (1 - overlap))

    # Calculate number of chunks
    n_chunks = ((len(data) - fs) // step) + 1

    # Get the number of features (columns) in the data
    n_features = data.shape[1]

    # Prepare output array with correct shape
    chunks = np.zeros((n_chunks, fs, n_features))

    # Fill chunks using sliding window
    for i in range(n_chunks):
        start_idx = i * step
        end_idx = start_idx + fs
        chunks[i] = data[start_idx:end_idx, :]

    return chunks
