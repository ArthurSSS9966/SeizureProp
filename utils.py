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
    Function to split data into chunks of size fs with specified overlap.
    Works with both 2D arrays (time, channel) and 3D arrays (time, channel, feature).

    Parameters:
    -----------
    data : numpy.ndarray
        Input data to be split, shape (time, channel) or (time, channel, feature)
    fs : int
        Window size (number of time steps per chunk)
    overlap : float
        Overlap between consecutive windows (0 to 1), default is 0
        e.g., 0.5 means 50% overlap

    Returns:
    --------
    numpy.ndarray
        Array of overlapped chunks with shape:
        - For 2D input: (n_chunks, fs, n_channels)
        - For 3D input: (n_chunks, fs, n_channels, n_features)
    '''
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")

    # Calculate step size based on overlap
    step = int(fs * (1 - overlap))

    # Calculate number of chunks
    n_chunks = ((len(data) - fs) // step) + 1

    # Determine if the input is 2D or 3D
    is_3d = len(data.shape) == 3

    if is_3d:
        # 3D data: (time, channel, feature)
        n_channels = data.shape[1]
        n_features = data.shape[2]

        # Prepare output array with correct shape for 3D
        chunks = np.zeros((n_chunks, fs, n_channels, n_features))

        # Fill chunks using sliding window
        for i in range(n_chunks):
            start_idx = i * step
            end_idx = start_idx + fs
            chunks[i] = data[start_idx:end_idx, :, :]
    else:
        # 2D data: (time, channel)
        n_channels = data.shape[1]

        # Prepare output array with correct shape for 2D
        chunks = np.zeros((n_chunks, fs, n_channels))

        # Fill chunks using sliding window
        for i in range(n_chunks):
            start_idx = i * step
            end_idx = start_idx + fs
            chunks[i] = data[start_idx:end_idx, :]

    return chunks

def parse_channel_string(channel_str):
    # Split the string by commas and strip whitespace
    channel_groups = [group.strip() for group in channel_str.split(',')]

    # Initialize list for all channels
    all_channels = []

    for group in channel_groups:
        # Split into prefix and range
        prefix = ''.join(c for c in group if not c.isdigit() and c != '-')
        # Get the range part
        range_part = group[len(prefix):]

        if '-' in range_part:
            # Get start and end numbers
            start, end = map(int, range_part.split('-'))
            # Generate all channel numbers in range
            for num in range(start, end + 1):
                all_channels.append(f"{prefix}{num}")
        else:
            # Single channel case
            all_channels.append(group)

    return all_channels


def find_seizure_channels(seizure_marking: pd.DataFrame, seizure_no: int, patient_no: int) -> list:
    """
    Find the channels that have seizure marking for a given seizure number

    Args:
    seizure_marking (pd.DataFrame): The seizure marking dataframe
    seizure_no (int): The seizure number
    seizure_no (int): The patient number

    Returns:
    list: The list of channels that have seizure marking
    """

    patient_marking = seizure_marking[seizure_marking['ID'] == patient_no]
    seizures = patient_marking[patient_marking['Seizure'] == seizure_no]

    channels = parse_channel_string(seizures['Seizure channels'].values[0])

    return channels


def find_seizure_onset(seizure_marking: pd.DataFrame, seizure_no: int, patient_no: int) -> list:
    """
    Find the seizure onset time for a given seizure number

    Args:
    seizure_marking (pd.DataFrame): The seizure marking dataframe
    seizure_no (int): The seizure number
    seizure_no (int): The patient number

    Returns:
    float: The seizure onset time
    """
    patient_marking = seizure_marking[seizure_marking['ID'] == patient_no]
    seizure_marking = patient_marking[patient_marking['Seizure'] == seizure_no]

    onset_time = seizure_marking['Seizure onset (1st second)'].values[0]

    return parse_channel_string(onset_time)

def find_seizure_related_channels(seizure_marking: pd.DataFrame, seizure_no: int, patient_no: int):
    """
    Find the channels that have seizure marking for a given seizure number

    Args:
    seizure_marking (pd.DataFrame): The seizure marking dataframe
    seizure_no (int): The seizure number
    seizure_no (int): The patient number

    Returns:
    list: The list of channels that have seizure marking
    """

    seizure_channels = find_seizure_channels(seizure_marking, seizure_no, patient_no)
    seizure_onset = find_seizure_onset(seizure_marking, seizure_no, patient_no)

    return seizure_channels, seizure_onset


def create_channel_mapping(gridmap_df):
    """
    Creates bidirectional dictionaries mapping channel numbers to names and vice versa

    Args:
        gridmap_df: DataFrame containing gridmap information with columns
                   'Label' and 'Channel'

    Returns:
        tuple: (num_to_name, name_to_num) dictionaries for bidirectional mapping
    """
    num_to_name = {}
    name_to_num = {}

    for _, row in gridmap_df.iterrows():
        label = row['Label']
        # Convert channel range string (e.g., "1:8") to list of integers
        chan_range = [int(x) for x in row['Channel'].split(':')]
        # Convert electrode range string to list of integers
        elec_range = [int(x) for x in row['GridElectrode'].split(':')]

        # Create mapping for each channel in the range
        for chan, elec in zip(range(chan_range[0], chan_range[1] + 1),
                              range(elec_range[0], elec_range[1] + 1)):
            channel_name = f"{label}{elec}"
            num_to_name[chan] = channel_name
            name_to_num[channel_name] = chan

    return num_to_name, name_to_num


def map_seizure_channels(channels, gridmap, mode='num_to_name'):
    """
    Maps between channel numbers and channel names

    Args:
        channels: List of channel numbers or names
        gridmap: DataFrame with gridmap information
        mode: 'num_to_name' to convert numbers to names,
              'name_to_num' to convert names to numbers

    Returns:
        list: Mapped channel names or numbers
    """
    num_to_name, name_to_num = create_channel_mapping(gridmap)

    # Convert input to list if it's not already
    channels = list(channels)

    if mode == 'num_to_name':
        mapping_dict = num_to_name
        unknown_format = "Unknown_{}"
    else:  # name_to_num
        mapping_dict = name_to_num
        unknown_format = -1  # Use -1 for unknown channel names

    # Map channels using appropriate dictionary
    mapped_channels = [mapping_dict.get(chan, unknown_format.format(chan) if mode == 'num_to_name' else unknown_format)
                       for chan in channels]

    return mapped_channels

def map_channels_to_numbers(channel_numbers, channel_names, selected_channels):
    """
    Maps selected channel names to their corresponding channel numbers, ignoring case.

    Parameters:
    channel_numbers (list): List of channel numbers
    channel_names (list): List of channel names corresponding to the numbers
    selected_channels (list): List of channel names to map back to numbers

    Returns:
    list: Channel numbers corresponding to the selected channel names

    Raises:
    ValueError: If channel_numbers and channel_names have different lengths
    ValueError: If a selected channel name is not found in channel_names
    """
    # Verify input lists have same length
    if len(channel_numbers) != len(channel_names):
        raise ValueError("Channel numbers and names lists must have the same length")

    # Create a dictionary mapping lowercase names to numbers
    channel_map = {name.lower(): num for name, num in zip(channel_names, channel_numbers)}

    # Map selected channels to numbers
    selected_numbers = []
    for channel in selected_channels:
        channel_lower = channel.lower()
        if channel_lower not in channel_map:
            raise ValueError(f"Channel name '{channel}' not found in channel_names")
        selected_numbers.append(channel_map[channel_lower])

    return selected_numbers