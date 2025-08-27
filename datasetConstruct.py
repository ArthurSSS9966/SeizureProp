import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import mne
from torch.utils.data import Dataset
import torch
from utils import (process_edf_channels_and_create_new_gridmap, find_seizure_annotations, split_data,
                   find_seizure_related_channels, map_seizure_channels)
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit

def constructDataset(data):
    '''
    This function constructs the dataset from the data file
    :param data: structure containing the fields "seizureNumber" and "data"
    :return: dataset structure containing the fields "seizureNumber", "units", "samplingRate", and "eegData"
    '''

    rawdata = data["data"]
    seizureNumber = data["seizureNumber"]

    # Process the txt file:

    # 1. Separate the section with a lot of % signs
    # 2. Extract the data from the section
    # 3. Save the data to a new file

    # Split the data by the % signs
    data = rawdata.split("%")

    # Remove empty sections
    data = [section for section in data if section != '']

    units = data[4].replace("\n", "").replace("\t", "").split(":")[-1]
    samplingRate = int(data[7].split()[2].split(".")[0])
    eegData = data[14]

    # Extract the data from eegData for each line
    eegData = eegData.split("\n")

    timeStamp = [line.split("\t")[0] for line in eegData][1:]
    # Remove all None values
    timeStamp = [ts for ts in timeStamp if ts != '' and ts != '--- BREAK IN DATA ---']

    # Convert the string timestamps to datetime objects
    datetime_objects = [datetime.strptime(ts, '%m/%d/%Y %H:%M:%S') for ts in timeStamp]

    # Define the start time (index 0 timestamp) and calculate the reference time
    start_time = datetime_objects[0]
    reference_time = start_time + timedelta(seconds=10)
    # Initialize an empty list for normalized times
    normalized_times = []

    # Calculate the normalized time in milliseconds relative to the reference time
    for i, dt in enumerate(datetime_objects):
        # Calculate the difference in seconds between the current time and the reference time
        time_diff_seconds = (dt - reference_time).total_seconds()

        # Determine the sample index within the second
        sample_index = i % samplingRate  # Assumes that samples are in order

        # Calculate the time in milliseconds for the sample within its second
        time_in_milliseconds = time_diff_seconds * 1000 + (sample_index * (1000 / samplingRate))

        normalized_times.append(time_in_milliseconds)

    normalized_times = np.array(normalized_times)

    # Separate the data by tabs, and excludes the first three tabs and the last tab
    eegData = [line.split("\t")[3:-1] for line in eegData][1:]

    # Convert the data to a DataFrame
    eegData = pd.DataFrame(eegData)

    # Delete the columns with "SHORT" value
    eegData = eegData.loc[:, (eegData != "SHORT").all()]

    # Replace the "AMPSAT" with maximum value
    eegData = eegData.replace("AMPSAT", 2000)

    # Delete the rows with None values
    eegData = eegData.dropna()

    # Convert the data to float
    eegData = np.array(eegData.astype(float))


    # Construct the structure of the dataset
    dataset = {
        "seizureNumber": seizureNumber,
        "units": units,
        "samplingRate": samplingRate,
        "normalized_times": normalized_times,
        "eegData": eegData
    }

    return dataset


def reconsEDF(raw, gridmap, PAT_NO):
    '''
    This function takes the .edf data and only extract the things that are necessary and save it to a new file
    :param raw:
    :return:
    '''

    filename = raw.annotations.description[0]

    # Find the correct file name in description that match SZXX where XX is the seizure number

    # Check if filename starts with SZ
    if not (filename.startswith("SZ") or filename.startswith("STIM")):
        for i, desc in enumerate(raw.annotations.description):
            if desc.startswith("SZ"):
                filename = desc
                break

    # Extract the data from the raw file
    data, time = raw[:]
    data = data.T
    samplingRate = int(raw.info["sfreq"])

    # Select useful channels based on gridmap
    channel_list = raw.ch_names
    gridmap, data = process_edf_channels_and_create_new_gridmap(gridmap, channel_list, data)

    # Check if STIMSZ is present in the filename
    if "STIMSZ" in filename:
        # Find the index of the STIMSZ where annotation contains STIMSZ
        SZON_ind = np.where(raw.annotations.description == "SZSTIMON")[0][0]
        SZOFF_ind = np.where(raw.annotations.description == 'SZSTIMOFF')[0][0]

    else:
        SZON_ind, SZOFF_ind = find_seizure_annotations(raw)

    SZ_time = int(raw.annotations.onset[SZON_ind] * samplingRate)
    SZOFF_time = int(raw.annotations.onset[SZOFF_ind] * samplingRate)

    # Define the preictal range
    pre_range = int(60*raw.info["sfreq"])  # 60 seconds before EOF and after SOF, also 10 seconds before Seizure

    # Extract the preictal and ictal data
    ictal_data = data[SZ_time:SZOFF_time, :]
    preictal_data2 = data[SZ_time-pre_range:SZ_time, :] if SZ_time-pre_range > 0 else data[:SZ_time-int(5*raw.info["sfreq"]), :]
    postictal_data2 = data[SZOFF_time:SZOFF_time+pre_range, :] if SZOFF_time+pre_range < data.shape[0] else data[SZOFF_time+int(5*raw.info["sfreq"]):, :]

    # Constrcut the dataset
    dataset = EDFData(gridmap, filename, PAT_NO)
    dataset.channelNumber = data.shape[1]
    dataset.samplingRate = samplingRate
    dataset.ictal = ictal_data
    dataset.interictal = preictal_data2
    dataset.postictal = postictal_data2
    dataset.annotations = raw.annotations.description
    dataset.annotations_onset = raw.annotations.onset
    dataset.channel_names = raw.ch_names
    # Add documentation to each variable
    dataset.documentation= ("ictal: Data during seizure\n,"
                            "preictal: 55-60 seconds free of seizure Data before seizure\n,"
                            "postictal: 55-60 seconds free of seizure Data after seizure\n,"
                            )

    return dataset


class EDFData:
    def __init__(self, gridmap, seizureNumber, patNo):
        self.samplingRate = None
        self.preictal2 = None
        self.ictal = None
        self.gridmap = gridmap
        self.seizureNumber = seizureNumber
        self.patNo = patNo

    def __str__(self):
        return f"Seizure {self.seizureNumber} for patient {self.patNo} data"

    def __repr__(self):
        return f"Seizure {self.seizureNumber} for patient {self.patNo} data"

    def __getitem__(self, item):
        if item == "gridmap":
            return self.gridmap
        elif item == "seizureNumber":
            return self.seizureNumber
        else:
            raise ValueError(f"Item {item} not found")


class CustomDataset(Dataset):
    def __init__(self, X, y=None):
        # Convert numpy array to PyTorch tensor
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_label = self.y[idx]  # Assuming self.y is defined in __init__
        return X_sample, y_label


def load_seizure(path, marking_file='data/Seizure_Onset_Type_ML_USC.xlsx'):
    """
    Load the seizure data from the specified path and store seizure channel and grey matter information.

    Args:
        path: Path to the seizure data
        marking_file: Path to the seizure marking file

    Returns:
        Combined seizure data with seizure_channels and grey_matter_mask attributes
    """
    seizure_data_combined = EDFData(None, None, None)
    raw_data_list = []  # Store all raw data for later processing

    # First pass: collect raw data
    cleaned_files = [f for f in os.listdir(path) if f.endswith("CLEANED.pkl") and ("STIM" not in f)]

    for cleaned_file in cleaned_files:
        raw = pickle.load(open(os.path.join(path, cleaned_file), "rb"))
        # Extract seizure number from filename
        # Pattern: "seizure_SZX_CLEANED.pkl" or similar
        import re
        match = re.search(r'seizure_SZ(\d+)_', cleaned_file)
        if match:
            seizure_No = int(match.group(1))
            raw.seizureNumber = f"SZ{seizure_No}"
        else:
            print(f"Warning: Could not extract seizure number from filename: {cleaned_file}")
            # Use default from the file if available, or set to unknown
            seizure_No = 1
            if hasattr(raw, 'seizureNumber') and raw.seizureNumber and isinstance(raw.seizureNumber, str):
                if raw.seizureNumber.startswith("SZ"):
                    try:
                        seizure_No = int(raw.seizureNumber[2:])
                    except ValueError:
                        pass
            raw.seizureNumber = f"SZ{seizure_No}"

        # Load seizure channel information
        try:
            # Load seizure marking data
            seizure_marking = pd.read_excel(marking_file)

            # Find seizure-related channels
            seizure_channels, _ = find_seizure_related_channels(
                seizure_marking, seizure_No, raw.patNo
            )
            # Store channel indices directly
            raw.seizure_channels = map_seizure_channels(seizure_channels, raw.gridmap, mode='name_to_num')
        except Exception as e:
            print(f"Warning: Couldn't get seizure channels for seizure {seizure_No}: {str(e)}")
            raw.seizure_channels = []

        # Load grey matter information (if available)
        try:
            # Load matter information from matter.csv if exists
            matter_file = os.path.join(path, "matter.csv")
            if os.path.exists(matter_file):
                matter = pd.read_csv(matter_file)

                # Get all electrode names
                all_channels = []
                for i in range(raw.ictal.shape[1]):
                    # Map channel number to name
                    channel_name = map_seizure_channels([i], raw.gridmap)[0]
                    all_channels.append(channel_name)

                # Create grey matter mask with float values instead of boolean
                # 1.0 = gray matter, 0.0 = white matter, 0.5 = ambiguous
                grey_matter_values = np.zeros(raw.ictal.shape[1], dtype=float)

                # For each channel, assign a value based on matter type
                for i, channel_name in enumerate(all_channels):
                    if channel_name in matter['ElectrodeName'].values:
                        # Get matter type
                        matter_type = matter[matter['ElectrodeName'] == channel_name]['MatterType'].values[0]
                        if matter_type == 'G':  # Grey matter
                            grey_matter_values[i] = 1.0
                        elif matter_type == 'W':  # White matter
                            grey_matter_values[i] = 0.0
                        elif matter_type == 'A':  # Ambiguous
                            grey_matter_values[i] = 0.5  # Intermediate value

                raw.grey_matter_values = grey_matter_values

                # You can also keep a boolean mask for compatibility
                grey_matter_mask = grey_matter_values > 0.0  # Treat both G and A as grey matter
                raw.grey_matter_mask = grey_matter_mask
            else:
                raw.grey_matter_values = None
                raw.grey_matter_mask = None

        except Exception as e:
            print(f"Warning: Couldn't get grey matter info for seizure {seizure_No}: {str(e)}")
            raw.grey_matter_mask = None

        raw_data_list.append(raw)

    # Process each seizure
    for i, raw in enumerate(raw_data_list):
        # Split the data into segments
        raw.interictal = split_data(raw.interictal, raw.samplingRate)
        raw.ictal = split_data(raw.ictal, raw.samplingRate)
        raw.postictal = split_data(raw.postictal, raw.samplingRate)

        # Initialize combined data with the first seizure
        if seizure_data_combined.patNo is None:
            seizure_data_combined = raw
            # delete the annotation, annotation_onset, and channel_names, and change the documentation
            del seizure_data_combined.annotations
            del seizure_data_combined.annotations_onset
            del seizure_data_combined.channel_names
            seizure_data_combined.documentation = ("Combined data from all seizures\n"
                                                   "ictal: Data during seizure\n"
                                                   "grey_matter_values: Grey matter values for each channel\n"
                                                   "grey_matter_mask: Grey matter mask for each channel\n"
                                                   "seizure_specific_channels: Dictionary mapping seizure numbers to channel indices\n"
                                                   "segment_to_seizure: Dictionary mapping segment indices to seizure numbers\n"
                                                   "seizure_segment_counts: Dictionary mapping seizure numbers to segment counts\n"
                                                   "patNo: Patient number\n")

            # Create dictionary to store seizure-specific channel information
            seizure_data_combined.seizure_specific_channels = {}
            seizure_data_combined.seizure_specific_channels[raw.seizureNumber] = raw.seizure_channels

            # Keep track of which segments belong to which seizure
            seizure_data_combined.segment_to_seizure = {}
            for seg_idx in range(len(raw.ictal)):
                seizure_data_combined.segment_to_seizure[seg_idx] = raw.seizureNumber

            # Track the number of segments from each seizure
            seizure_data_combined.seizure_segment_counts = {raw.seizureNumber: len(raw.ictal)}
        else:
            # Keep track of current segment count to use as offset for new segments
            prev_segments_count = len(seizure_data_combined.ictal)

            # Check if dimensions match and adjust if needed
            if raw.ictal.shape[2] != seizure_data_combined.ictal.shape[2]:
                print(f"Warning: Channel mismatch in seizure {raw.seizureNumber}. Adjusting dimensions...")
                # Pad with zeros if necessary
                max_channels = max(raw.ictal.shape[2], seizure_data_combined.ictal.shape[2])

                # Pad current seizure if needed
                if raw.ictal.shape[2] < max_channels:
                    padding = np.zeros((raw.ictal.shape[0], raw.ictal.shape[1],
                                        max_channels - raw.ictal.shape[2]))
                    raw.ictal = np.concatenate((raw.ictal, padding), axis=2)

                # Pad combined data if needed
                if seizure_data_combined.ictal.shape[2] < max_channels:
                    padding = np.zeros((seizure_data_combined.ictal.shape[0],
                                        seizure_data_combined.ictal.shape[1],
                                        max_channels - seizure_data_combined.ictal.shape[2]))
                    seizure_data_combined.ictal = np.concatenate((seizure_data_combined.ictal,
                                                                  padding), axis=2)

            # Combine the data
            seizure_data_combined.ictal = np.vstack((seizure_data_combined.ictal, raw.ictal))
            seizure_data_combined.interictal = np.vstack((seizure_data_combined.interictal,
                                                          raw.interictal))
            seizure_data_combined.postictal = np.vstack((seizure_data_combined.postictal,
                                                         raw.postictal))

            # Store seizure-specific channel information
            seizure_data_combined.seizure_specific_channels[raw.seizureNumber] = raw.seizure_channels

            # Keep track of which segments belong to which seizure
            for seg_idx in range(len(raw.ictal)):
                global_seg_idx = prev_segments_count + seg_idx
                seizure_data_combined.segment_to_seizure[global_seg_idx] = raw.seizureNumber

            # Track the segment counts
            seizure_data_combined.seizure_segment_counts[raw.seizureNumber] = len(raw.ictal)

    seizure_data_combined.seizureNumber = 'All'

    # Save the combined data
    with open(os.path.join(path, "seizure_All_combined.pkl"), "wb") as f:
        pickle.dump(seizure_data_combined, f)

    return seizure_data_combined


def load_seizure_across_patients(data_folder):

    seizure_data_combined = []

    for folder in os.listdir(data_folder):
        # if not a folder or the folder does not start with P, skip
        if not os.path.isdir(os.path.join(data_folder, folder)) or not folder.startswith("P"):
            continue
        # if there is a file called seizure_combined
        if os.path.exists(os.path.join(data_folder, folder, "seizure_All_combined.pkl")):
            seizure_single = pickle.load(open(os.path.join(data_folder, folder, "seizure_All_combined.pkl"), "rb"))
            seizure_data_combined.append(seizure_single)
        else:
            seizure_single = load_seizure(os.path.join(data_folder, folder))
            seizure_data_combined.append(seizure_single)

    return seizure_data_combined


def load_single_seizure(path, seizure_number):
    """
    Load the seizure data from the specified path and seizure number.

    :param path: Path to the seizure data
    :param seizure_number: Seizure number to load
    :return: Tuple of (raw, gridmap)
    """

    if os.path.exists(os.path.join(path, f"seizure_SZ{seizure_number}_combined.pkl")):
        raw = pickle.load(open(os.path.join(path, f"seizure_SZ{seizure_number}_combined.pkl"), "rb"))
    else:
        # Find all seizure data that ends with CLEANED.pkl and does not have STIM in the name
        cleaned_file = [f for f in os.listdir(path) if f.endswith("CLEANED.pkl") and f.startswith(f"seizure_SZ{seizure_number}")]

        raw = pickle.load(open(os.path.join(path, cleaned_file[0]), "rb"))

        raw.interictal = split_data(raw.interictal, raw.samplingRate)
        raw.ictal = split_data(raw.ictal, raw.samplingRate)
        raw.postictal = split_data(raw.postictal, raw.samplingRate)

    # Load seizure channel information
    try:
        # Load seizure marking data
        seizure_marking = pd.read_excel('data/Seizure_Onset_Type_ML_USC.xlsx')

        # Find seizure-related channels
        seizure_channels, _ = find_seizure_related_channels(
            seizure_marking, seizure_number, raw.patNo
        )
        # Store channel indices directly
        raw.seizure_channels = map_seizure_channels(seizure_channels, raw.gridmap, mode='name_to_num')
    except Exception as e:
        print(f"Warning: Couldn't get seizure channels for seizure {seizure_number}: {str(e)}")
        raw.seizure_channels = []

    # Load grey matter information (if available)
    try:
        # Load matter information from matter.csv if exists
        matter_file = os.path.join(path, "matter.csv")
        if os.path.exists(matter_file):
            matter = pd.read_csv(matter_file)

            # Get all electrode names
            all_channels = []
            for i in range(raw.ictal.shape[1]):
                # Map channel number to name
                channel_name = map_seizure_channels([i], raw.gridmap)[0]
                all_channels.append(channel_name)

            # Create grey matter mask with float values instead of boolean
            # 1.0 = gray matter, 0.0 = white matter, 0.5 = ambiguous
            grey_matter_values = np.zeros(raw.ictal.shape[1], dtype=float)

            # For each channel, assign a value based on matter type
            for i, channel_name in enumerate(all_channels):
                if channel_name in matter['ElectrodeName'].values:
                    # Get matter type
                    matter_type = matter[matter['ElectrodeName'] == channel_name]['MatterType'].values[0]
                    if matter_type == 'G':  # Grey matter
                        grey_matter_values[i] = 1.0
                    elif matter_type == 'W':  # White matter
                        grey_matter_values[i] = 0.0
                    elif matter_type == 'A':  # Ambiguous
                        grey_matter_values[i] = 0.5  # Intermediate value

            raw.grey_matter_values = grey_matter_values

            # You can also keep a boolean mask for compatibility
            grey_matter_mask = grey_matter_values > 0.0  # Treat both G and A as grey matter
            raw.grey_matter_mask = grey_matter_mask
        else:
            raw.grey_matter_values = None
            raw.grey_matter_mask = None

    except Exception as e:
        print(f"Warning: Couldn't get grey matter info for seizure {seizure_number}: {str(e)}")
        raw.grey_matter_mask = None

    return raw


def create_dataset(seizure, train_percentage=0.8, batch_size=512, min_activity_threshold=1e-6,
                   input_type='raw', window_size=20, sliding_step=5):
    """
    Create training and validation datasets with proper masking for all input types.

    Args:
        seizure: Seizure data object containing ictal, interictal and postictal data
        train_percentage: Percentage of data to use for training
        batch_size: Size of batches for training
        min_activity_threshold: Minimum activity threshold to consider a sample non-empty
        input_type: Type of input data to use ('raw', 'transformed', or 'combined')
        window_size: Number of consecutive windows to include for transformed features
        sliding_step: Step size for sliding window in transformed features

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """

    # Define custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data, labels, channel_idx, time_idx, seizure_mask=None, grey_matter_values=None):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.channel_idx = torch.tensor(channel_idx, dtype=torch.long)
            self.time_idx = torch.tensor(time_idx, dtype=torch.long)
            self.seizure_mask = torch.tensor(seizure_mask, dtype=torch.bool) if seizure_mask is not None else None
            self.grey_matter_values = torch.tensor(grey_matter_values, dtype=torch.float32) if grey_matter_values is not None else None

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = {
                'data': self.data[idx],
                'label': self.labels[idx],
                'channel_idx': self.channel_idx[idx],
                'time_idx': self.time_idx[idx]
            }
            if self.seizure_mask is not None:
                item['seizure_mask'] = self.seizure_mask[idx]
            if self.grey_matter_values is not None:
                item['grey_matter_values'] = self.grey_matter_values[idx]
            return item

    # Check seizure-specific info
    has_seizure_specific = (
            hasattr(seizure, 'seizure_specific_channels') and
            bool(seizure.seizure_specific_channels) and
            hasattr(seizure, 'segment_to_seizure')
    )
    has_grey_matter = hasattr(seizure, 'grey_matter_values') and seizure.grey_matter_values is not None

    if input_type == 'raw':
        seizure_data = seizure.ictal.transpose(0, 2, 1)
        nonseizure_data = np.concatenate((seizure.interictal, seizure.postictal), axis=0).transpose(0, 2, 1)

        data_list = []
        label_list = []
        channel_idx_list = []
        time_idx_list = []
        seizure_mask_list = []
        grey_matter_list = []

        # Process seizure segments
        for seg_idx, segment in enumerate(seizure_data):
            seizure_num = seizure.segment_to_seizure.get(seg_idx, 'All') if has_seizure_specific else 'All'
            seizure_channels = seizure.seizure_specific_channels.get(seizure_num, []) if has_seizure_specific else []
            for ch_idx, signal in enumerate(segment):
                if np.max(np.abs(signal)) > min_activity_threshold:
                    data_list.append(signal)
                    label_list.append(1)
                    channel_idx_list.append(ch_idx)
                    time_idx_list.append(seg_idx)
                    seizure_mask_list.append(1 if ch_idx in seizure_channels else 0)
                    if has_grey_matter:
                        grey_matter_list.append(seizure.grey_matter_values[ch_idx])

        # Process non-seizure segments
        offset_seg_idx = seizure_data.shape[0]  # Nonseizure segments are after ictal
        for seg_idx, segment in enumerate(nonseizure_data):
            for ch_idx, signal in enumerate(segment):
                if np.max(np.abs(signal)) > min_activity_threshold:
                    data_list.append(signal)
                    label_list.append(0)
                    channel_idx_list.append(ch_idx)
                    time_idx_list.append(offset_seg_idx + seg_idx)
                    seizure_mask_list.append(0)
                    if has_grey_matter:
                        grey_matter_list.append(seizure.grey_matter_values[ch_idx])

    elif input_type == 'transformed':
        seizure_data = seizure.ictal_transformed
        nonseizure_data = np.concatenate((seizure.interictal_transformed, seizure.postictal_transformed), axis=0)

        data_list = []
        label_list = []
        channel_idx_list = []
        time_idx_list = []
        seizure_mask_list = []
        grey_matter_list = []

        # Process seizure segments
        for seg_idx, segment in enumerate(seizure_data):
            seizure_num = seizure.segment_to_seizure.get(seg_idx, 'All') if has_seizure_specific else 'All'
            seizure_channels = seizure.seizure_specific_channels.get(seizure_num, []) if has_seizure_specific else []

            for ch_idx, signal in enumerate(segment):
                if np.max(np.abs(signal)) > min_activity_threshold:
                    signal = signal.transpose(1, 0)  # (features, time_steps)
                    data_list.append(signal)
                    label_list.append(1)
                    channel_idx_list.append(ch_idx)
                    time_idx_list.append(seg_idx)
                    grey_matter_list.append(seizure.grey_matter_values[ch_idx])

                    # seizure mask
                    if ch_idx in seizure_channels:
                        seizure_mask_list.append(1)
                    else:
                        seizure_mask_list.append(0)

        # Process non-seizure segments
        offset_seg_idx = seizure_data.shape[0]
        for seg_idx, segment in enumerate(nonseizure_data):
            for ch_idx, signal in enumerate(segment):
                if np.max(np.abs(signal)) > min_activity_threshold:
                    signal = signal.transpose(1, 0)  # (features, time_steps)
                    data_list.append(signal)
                    label_list.append(0)
                    channel_idx_list.append(ch_idx)
                    time_idx_list.append(seg_idx)
                    seizure_mask_list.append(0)
                    grey_matter_list.append(0)

    else:
        raise ValueError("Invalid input_type. Choose 'raw' or 'transformed'.")

    # Convert to numpy arrays
    X = np.array(data_list)
    y = np.array(label_list)
    channel_idx = np.array(channel_idx_list)
    time_idx = np.array(time_idx_list)
    seizure_mask = np.array(seizure_mask_list)
    grey_values = np.array(grey_matter_list) if grey_matter_list else None

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    channel_idx = channel_idx[indices]
    time_idx = time_idx[indices]
    seizure_mask = seizure_mask[indices]
    if grey_values is not None:
        grey_values = grey_values[indices]

    # Balanced sample (optional)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_samples = min(len(pos_idx), len(neg_idx))
    pos_idx = np.random.choice(pos_idx, n_samples, replace=False)
    neg_idx = np.random.choice(neg_idx, n_samples, replace=False)

    balanced_idx = np.concatenate([pos_idx, neg_idx])
    X = X[balanced_idx]
    y = y[balanced_idx]
    channel_idx = channel_idx[balanced_idx]
    time_idx = time_idx[balanced_idx]
    seizure_mask = seizure_mask[balanced_idx]
    if grey_values is not None:
        grey_values = grey_values[balanced_idx]

    # Train/Val split
    n_train = int(train_percentage * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    channel_idx_train, channel_idx_val = channel_idx[:n_train], channel_idx[n_train:]
    time_idx_train, time_idx_val = time_idx[:n_train], time_idx[n_train:]
    seizure_mask_train, seizure_mask_val = seizure_mask[:n_train], seizure_mask[n_train:]

    if grey_values is not None:
        grey_train, grey_val = grey_values[:n_train], grey_values[n_train:]
    else:
        grey_train = grey_val = None

    train_dataset = CustomDataset(X_train, y_train, channel_idx_train, time_idx_train, seizure_mask_train, grey_train)
    val_dataset = CustomDataset(X_val, y_val, channel_idx_val, time_idx_val, seizure_mask_val, grey_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Final dataset shapes - Train: {X_train.shape}, Val: {X_val.shape}")

    return train_loader, val_loader

def combine_loaders(loader_list, batch_size=512):
    # Extract datasets from DataLoaders
    train_datasets = [loader[0].dataset for loader in loader_list]  # Get datasets from training DataLoaders
    val_datasets = [loader[1].dataset for loader in loader_list]  # Get datasets from validation DataLoaders

    # Combine training datasets
    combined_train = ConcatDataset(train_datasets)
    # Combine validation datasets
    combined_val = ConcatDataset(val_datasets)

    # Create new loaders
    train_loader = DataLoader(
        dataset=combined_train,
        batch_size=batch_size,  # Adjust as needed
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=combined_val,
        batch_size=batch_size,  # Adjust as needed
        shuffle=False  # Usually False for validation
    )

    return train_loader, val_loader


def construct_single_dataset(results_propagation, window_size, overlap=0.5):
    """
    Constructs dataset by sliding window over smoothed probabilities to create N-second segments.

    Args:
        results_propagation: Dictionary containing smoothed_probabilities and true_seizure_channels
        window_size: Size of window in seconds
        overlap: Overlap fraction between consecutive windows (default 0.5 for 50% overlap)

    Returns:
        X: Array of resampled segments
        y: Array of corresponding labels
    """
    # Get raw data excluding first 50 timepoints
    raw_data = results_propagation['smoothed_probabilities'][50:, :]

    # Calculate window parameters
    window_length = window_size * 5  # 5 samples per second

    # Check if data is long enough for at least one window
    if len(raw_data) < window_length:
        print(f"Warning: Data length {len(raw_data)} is shorter than window length {window_length}")
        # Pad with zeros if needed
        padding = np.zeros((window_length - len(raw_data), raw_data.shape[1]))
        raw_data = np.vstack([raw_data, padding])

    step_size = int(window_length * (1 - overlap))  # Step size based on overlap

    # Ensure at least one complete segment
    if step_size == 0:
        step_size = 1

    # Calculate number of complete segments
    num_segments = (len(raw_data) - window_length) // step_size + 1

    # If no complete segments possible, create just one segment
    if num_segments <= 0:
        num_segments = 1

    # Initialize arrays
    X = np.zeros((num_segments, window_length, raw_data.shape[1]))
    y = np.zeros((num_segments, raw_data.shape[1]))

    # Create segments using sliding window
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + window_length

        # Handle case where end_idx exceeds data length
        if end_idx > len(raw_data):
            # Pad with zeros
            segment = raw_data[start_idx:, :]
            padding = np.zeros((end_idx - len(raw_data), raw_data.shape[1]))
            X[i] = np.vstack([segment, padding])
        else:
            # Extract segment normally
            X[i] = raw_data[start_idx:end_idx, :]

        # Assign labels
        y[i][results_propagation['true_seizure_channels']] = 1

    X = np.transpose(X, (0, 2, 1))
    # Combine the first two dimensions
    X = np.concatenate([X[i] for i in range(X.shape[0])], axis=0)
    y = y.flatten()

    # print the shape of X

    return X, y, X, y


def construct_channel_recognition_dataset(results_propagation_total, seizure_onset_window_size=30, batch_size=64,
                                          split=0.8, data_aug = False, **kwargs):
    """
    Constructs balanced datasets for seizure channel recognition and onset detection.

    Args:
        results_propagation_total: List of dictionaries containing seizure data
        seizure_onset_window_size: Window size in seconds (default: 60)
        batch_size: Size of training batches (default: 64)
        split: Train/validation split ratio (default: 0.8)

    Returns:
        Four DataLoaders: train/val for channel recognition and train/val for onset detection
    """

    def process_dataset(x_data, y_data, split, batch_size):
        """Helper function to reduce code duplication"""
        # Balance dataset
        pos_idx = np.where(y_data == 1)[0]
        neg_idx = np.where(y_data == 0)[0]
        n_samples = min(len(pos_idx), len(neg_idx))

        # Sample equal numbers
        indices = np.concatenate([
            np.random.choice(pos_idx, n_samples, replace=False),
            np.random.choice(neg_idx, n_samples, replace=False)
        ])
        indices = np.random.permutation(indices)

        x_data, y_data = x_data[indices], y_data[indices]

        # Split data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - split, random_state=0)
        train_idx, val_idx = next(sss.split(x_data, y_data))

        # Create datasets and loaders
        train_loader = DataLoader(
            CustomDataset(x_data[train_idx], y_data[train_idx]),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            CustomDataset(x_data[val_idx], y_data[val_idx]),
            batch_size=1,
            shuffle=True
        )

        return train_loader, val_loader

    # Collect data from all samples
    x_total, y_total = [], []
    x_onset_total, y_onset_total = [], []

    for results in results_propagation_total:
        x, y, x_onset, y_onset = construct_single_dataset(
            results,
            seizure_onset_window_size
        )
        x_total.append(x)
        y_total.append(y)
        x_onset_total.append(x_onset)
        y_onset_total.append(y_onset)

    # Process channel recognition data
    x_total = np.expand_dims(np.concatenate(x_total, axis=0), axis=1)
    y_total = np.concatenate(y_total)

    channel_train, channel_val = process_dataset(
        x_total, y_total, split, batch_size
    )

    # Process onset detection data
    x_onset_total = np.expand_dims(np.concatenate(x_onset_total, axis=0), axis=1)
    y_onset_total = np.concatenate(y_onset_total)

    onset_train, onset_val = process_dataset(
        x_onset_total, y_onset_total, split, batch_size
    )

    return channel_train, channel_val, onset_train, onset_val





if __name__ == "__main__":
    DATA_FOLDER = "D:/Blcdata/seizure"
    OUTPUT_FOLDER = "data"
    PAT_NOs = [62, 65, 66]

    for PAT_NO in PAT_NOs:
        # Load the data
        data_folder = os.path.join(DATA_FOLDER, "P0{:02d}".format(PAT_NO))

        # Load .txt grid map file
        gridmap_loc = f"gridmap.P0{PAT_NO}.txt"
        gridmap_loc = os.path.join(data_folder, gridmap_loc)
        gridmap = pd.read_csv(gridmap_loc, sep=",")

        # Load the edf data file
        seizurefiles = [f for f in os.listdir(data_folder) if f.lower().endswith(".edf")]

        for file in seizurefiles:

            # try:

            data = mne.io.read_raw_edf(os.path.join(data_folder, file))

            dataset = reconsEDF(data, gridmap, PAT_NO)
            output_dir = os.path.join(OUTPUT_FOLDER, f"P{PAT_NO}")

            # Save the data to a new file
            if not os.path.exists(output_dir):
                print(f"Creating directory {output_dir}")
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, f"seizure_{dataset['seizureNumber']}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(dataset, f)
                print(f"Data for seizure {dataset['seizureNumber']} saved to {output_file}")

            # except Exception as e:
            #     print(f"Error processing file {file}: {e}")
            #     continue