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
    preictal_range = int(60*raw.info["sfreq"])  # 60 seconds before EOF and after SOF, also 60 seconds before Seizure
    pre_range = int(10*raw.info["sfreq"])  # 10 seconds before EOF and after SOF, also 10 seconds before Seizure
    postictal_range = int(np.min([60*raw.info["sfreq"], data.shape[0] - SZOFF_time]))  # 60 seconds after EOF and before SOF

    # Extract the preictal and ictal data
    preictal_data = data[:preictal_range, :]
    postictal_data = data[postictal_range:, :]
    ictal_data = data[SZ_time:SZOFF_time, :]
    preictal_data2 = data[SZ_time-pre_range:SZ_time, :]
    postictal_data2 = data[SZOFF_time:SZOFF_time+pre_range, :]

    # Constrcut the dataset
    dataset = EDFData(gridmap, filename, PAT_NO)
    dataset.channelNumber = data.shape[1]
    dataset.samplingRate = samplingRate
    dataset.ictal = ictal_data
    dataset.interictal = preictal_data
    dataset.postictal = postictal_data
    dataset.preictal2 = preictal_data2
    dataset.postictal2 = postictal_data2
    dataset.annotations = raw.annotations.description
    dataset.annotations_onset = raw.annotations.onset
    dataset.channel_names = raw.ch_names
    # Add documentation to each variable
    dataset.documentation= ("ictal: Data during seizure\n,"
                            "preictal: 60 seconds free of seizure Data before seizure\n,"
                            "postictal: 60 seconds free of seizure Data after seizure\n,"
                            "preictal2: 10 seconds Data right before seizure\n,"
                            "postictal2: 10 seconds Data right after seizure"
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


def load_seizure(path):
    """
    Load the seizure data from the specified path and handle both marked and unmarked channels.

    :param path: Path to the seizure data
    :return: Combined seizure data
    """

    def load_seizure_data(seizure_no, patient_no, gridmap,
                          marking_file='data/Seizure_Onset_Type_ML_USC.xlsx'):
        """Load seizure data and channel information"""
        try:
            # Load seizure marking data
            seizure_marking = pd.read_excel(marking_file)

            # Find seizure-related channels
            seizure_channels, _ = find_seizure_related_channels(
                seizure_marking, seizure_no, patient_no
            )
            seizure_channels = map_seizure_channels(seizure_channels, gridmap, mode='name_to_num')
            return seizure_channels
        except:
            return None

    seizure_data_combined = EDFData(None, None, None)
    raw_data_list = []  # Store all raw data for later processing

    # First pass: collect raw data
    cleaned_files = [f for f in os.listdir(path) if f.endswith("CLEANED.pkl") and ("STIM" not in f)]

    for cleaned_file in cleaned_files:
        raw = pickle.load(open(os.path.join(path, cleaned_file), "rb"))
        seizure_No = int(raw.seizureNumber.split("SZ")[-1]) if raw.seizureNumber.split("SZ")[-1].isdigit() else 1
        seizure_channels = load_seizure_data(seizure_No, raw.patNo, raw.gridmap)

        # If no channels are marked, use all available channels
        if seizure_channels is None or len(seizure_channels) == 0:
            print(f"Warning: No marked channels for seizure {seizure_No} of patient {raw.patNo}. Using all channels.")
            seizure_channels = list(range(raw.ictal.shape[1]))  # Use all available channels

        raw_data_list.append((raw, seizure_channels))

    # Process each seizure
    for i, (raw, seizure_channels) in enumerate(raw_data_list):
        # Keep only the relevant channels for marked seizures
        if seizure_channels is not None:
            raw.ictal = raw.ictal[:, seizure_channels]

        # Split the data into segments
        raw.interictal = split_data(raw.interictal, raw.samplingRate)
        raw.ictal = split_data(raw.ictal, raw.samplingRate)
        raw.postictal = split_data(raw.postictal, raw.samplingRate)
        raw.preictal2 = split_data(raw.preictal2, raw.samplingRate)
        raw.postictal2 = split_data(raw.postictal2, raw.samplingRate)

        # Initialize combined data with the first seizure
        if seizure_data_combined.patNo is None:
            seizure_data_combined = raw
            seizure_data_combined.interictal = np.concatenate((seizure_data_combined.interictal,
                                                               raw.preictal2), axis=0)
            seizure_data_combined.postictal = np.concatenate((seizure_data_combined.postictal,
                                                              raw.postictal2), axis=0)
        else:
            # Check if dimensions match
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
                                                          raw.interictal, raw.preictal2))
            seizure_data_combined.postictal = np.vstack((seizure_data_combined.postictal,
                                                         raw.postictal, raw.postictal2))

    seizure_data_combined.seizureNumber = 'All'
    return seizure_data_combined

def load_seizure_across_patients(data_folder):

    seizure_data_combined = []

    for folder in os.listdir(data_folder):
        # if not a folder or the folder does not start with P, skip
        if not os.path.isdir(os.path.join(data_folder, folder)) or not folder.startswith("P"):
            continue
        # if there is a file called seizure_combined
        if os.path.exists(os.path.join(data_folder, folder, "seizure_combined.pkl")):
            seizure_single = pickle.load(open(os.path.join(data_folder, folder, "seizure_combined.pkl"), "rb"))
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

    # Find all seizure data that ends with CLEANED.pkl and does not have STIM in the name
    cleaned_file = [f for f in os.listdir(path) if f.endswith("CLEANED.pkl") and f.startswith(f"seizure_SZ{seizure_number}")]

    raw = pickle.load(open(os.path.join(path, cleaned_file[0]), "rb"))

    raw.interictal = split_data(raw.interictal, raw.samplingRate)
    raw.ictal = split_data(raw.ictal, raw.samplingRate)
    raw.postictal = split_data(raw.postictal, raw.samplingRate)

    return raw


def create_dataset(seizure, train_percentage=0.8, batch_size=512, min_activity_threshold=1e-6,
                   input_type='raw', window_size=20, sliding_step=5):
    """
    Create training and validation datasets, handling zero-padded channels after flattening.
    Supports raw data, transformed features, or a combined approach.

    Args:
        seizure: Seizure data object containing ictal, interictal, and postictal data
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

    # Define custom dataset class if not already defined
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    if input_type == 'raw':
        # Original raw data approach
        seizure_data = seizure.ictal
        nonseizure_data = seizure.interictal
        nonseizure_data_postictal = seizure.postictal

        # Combine the nonseizure and postictal data
        nonseizure_data = np.concatenate((nonseizure_data, nonseizure_data_postictal), axis=0)

        # Transpose to get the correct shape for the model
        seizure_data_flat = seizure_data.transpose(0, 2, 1)
        nonseizure_data_flat = nonseizure_data.transpose(0, 2, 1)

        # Flatten the dataset from [Sample, Channel, Time] to [Sample * Channel, Time, 1]
        seizure_data_flat = np.concatenate([seizure_data_flat[:, i, :] for i in range(seizure_data_flat.shape[1])],
                                           axis=0)
        nonseizure_data_flat = np.concatenate(
            [nonseizure_data_flat[:, i, :] for i in range(nonseizure_data_flat.shape[1])], axis=0)

        # Remove flattened samples that are all zeros (from padding)
        sample_activity = np.sum(np.abs(seizure_data_flat), axis=1)  # Sum across time dimension only
        active_samples = sample_activity.squeeze() > min_activity_threshold

        if np.sum(active_samples) == 0:
            print("Warning: No active samples found in seizure data. Check your data or threshold.")
        else:
            n_removed = len(active_samples) - np.sum(active_samples)
            print(f"Removed {n_removed} zero-padded samples out of {len(active_samples)} total samples")
            seizure_data_flat = seizure_data_flat[active_samples]

        # Create the labels
        seizure_labels = np.ones(len(seizure_data_flat))
        nonseizure_labels = np.zeros(len(nonseizure_data_flat))

        # Combine the dataset and labels
        data = np.concatenate((seizure_data_flat, nonseizure_data_flat), axis=0)
        data = np.expand_dims(data, axis=2)  # Add channel dimension
        data = np.transpose(data, (0, 2, 1))  # Transpose to [Sample, Channel, Time]
        labels = np.concatenate((seizure_labels, nonseizure_labels), axis=0)

    elif input_type == 'transformed':
        # Process transformed data
        # Check if the transformed data exists
        if not hasattr(seizure, 'ictal_transformed') or not hasattr(seizure, 'interictal_transformed'):
            raise ValueError("Transformed data not found. Run extract_features_store_in_object first.")

        # Prepare transformed data for LSTM/CNN input
        X_sequences = []
        y_labels = []

        # Process ictal data (seizure)
        ictal_data = seizure.ictal_transformed
        n_segments, n_channels, n_windows, n_features = ictal_data.shape

        if n_windows < window_size:
            print(f"Warning: ictal data has only {n_windows} windows, but window_size is {window_size}.")
        else:
            for segment_idx in range(n_segments):
                for channel_idx in range(n_channels):
                    # Get feature time series for this segment and channel
                    feature_array = ictal_data[segment_idx, channel_idx]

                    # Add to lists
                    X_sequences.append(feature_array)
                    y_labels.append(1)  # Seizure label

        # Process interictal data (non-seizure)
        interictal_data = seizure.interictal_transformed
        n_segments, n_channels, n_windows, n_features = interictal_data.shape

        if n_windows < window_size:
            print(f"Warning: interictal data has only {n_windows} windows, but window_size is {window_size}.")
        else:
            for segment_idx in range(n_segments):
                for channel_idx in range(n_channels):
                    # Get feature time series for this segment and channel
                    feature_array = interictal_data[segment_idx, channel_idx]

                    # Add to lists
                    X_sequences.append(feature_array)
                    y_labels.append(0)  # Non-seizure label

        # Process postictal data (also non-seizure but post-seizure)
        postictal_data = seizure.postictal_transformed
        n_segments, n_channels, n_windows, n_features = postictal_data.shape

        if n_windows < window_size:
            print(f"Warning: postictal data has only {n_windows} windows, but window_size is {window_size}.")
        else:
            for segment_idx in range(n_segments):
                for channel_idx in range(n_channels):
                    # Get feature time series for this segment and channel
                    feature_array = postictal_data[segment_idx, channel_idx]

                    # Add to lists
                    X_sequences.append(feature_array)
                    y_labels.append(0)  # Non-seizure label (postictal)

        # Convert to numpy arrays
        data = np.array(X_sequences)
        labels = np.array(y_labels)

        # For compatibility with the model, ensure data is shaped as [Samples, Channels, Time]
        # In this case, our "channels" dimension is the features dimension
        # Reshape from [Samples, Window_size, Features] to [Samples, Features, Window_size]
        data = np.transpose(data, (0, 2, 1))

    elif input_type == 'combined':
        # Combine raw data and transformed features
        # First process raw data
        seizure_data = seizure.ictal
        nonseizure_data = np.concatenate((seizure.interictal, seizure.postictal), axis=0)

        # Transpose raw data
        seizure_data_raw = seizure_data.transpose(0, 2, 1)
        nonseizure_data_raw = nonseizure_data.transpose(0, 2, 1)

        # Flatten raw data
        n_seizure_channels = seizure_data_raw.shape[1]
        n_nonseizure_channels = nonseizure_data_raw.shape[1]

        seizure_data_flat = np.concatenate([seizure_data_raw[:, i, :] for i in range(n_seizure_channels)], axis=0)
        nonseizure_data_flat = np.concatenate([nonseizure_data_raw[:, i, :] for i in range(n_nonseizure_channels)],
                                              axis=0)

        # Remove inactive samples from raw data
        sample_activity = np.sum(np.abs(seizure_data_flat), axis=1)
        active_samples = sample_activity.squeeze() > min_activity_threshold

        if np.sum(active_samples) == 0:
            print("Warning: No active samples found in seizure data. Check your data or threshold.")
        else:
            n_removed = len(active_samples) - np.sum(active_samples)
            print(f"Removed {n_removed} zero-padded samples out of {len(active_samples)} total samples")
            seizure_data_flat = seizure_data_flat[active_samples]

        # Now process transformed data
        X_sequences = []
        raw_segments = []
        y_labels = []

        # Process ictal data (seizure)
        ictal_transformed = seizure.ictal_transformed
        ictal_raw = seizure.ictal
        n_segments, n_channels, n_windows, n_features = ictal_transformed.shape

        for segment_idx in range(n_segments):
            for channel_idx in range(n_channels):
                # Get feature time series
                feature_array = ictal_transformed[segment_idx, channel_idx]

                # Get corresponding raw data
                raw_data = ictal_raw[segment_idx, :, channel_idx]

                # Skip if not enough windows
                if n_windows < window_size:
                    continue

                # Create sequences
                for i in range(0, n_windows - window_size + 1, sliding_step):
                    sequence = feature_array[i:i + window_size]
                    X_sequences.append(sequence)
                    raw_segments.append(raw_data)
                    y_labels.append(1)  # Seizure label

        # Process interictal and postictal data
        for state, data_source in [("interictal", seizure.interictal), ("postictal", seizure.postictal)]:
            transformed_data = getattr(seizure, f"{state}_transformed")
            raw_data = data_source
            n_segments, n_channels, n_windows, n_features = transformed_data.shape

            for segment_idx in range(n_segments):
                for channel_idx in range(n_channels):
                    # Get feature time series
                    feature_array = transformed_data[segment_idx, channel_idx]

                    # Get corresponding raw data
                    segment_raw = raw_data[segment_idx, :, channel_idx]

                    # Skip if not enough windows
                    if n_windows < window_size:
                        continue

                    # Create sequences
                    for i in range(0, n_windows - window_size + 1, sliding_step):
                        sequence = feature_array[i:i + window_size]
                        X_sequences.append(sequence)
                        raw_segments.append(segment_raw)
                        y_labels.append(0)  # Non-seizure label

        # Convert to numpy arrays
        feature_data = np.array(X_sequences)
        raw_data = np.array(raw_segments)
        labels = np.array(y_labels)

        # Reshape for model compatibility
        feature_data = np.transpose(feature_data, (0, 2, 1))  # [Samples, Features, Window_size]
        raw_data = np.expand_dims(raw_data, axis=1)  # [Samples, 1, Time]

        # Combine data (this approach concatenates along the channel dimension)
        # You might need to adjust depending on your model architecture
        # For example, you might need a different approach like having two inputs
        # This is a simplistic approach assuming your model can handle varying channel counts
        data = np.concatenate([raw_data, feature_data], axis=1)

    else:
        raise ValueError("Invalid input_type. Choose 'raw', 'transformed', or 'combined'.")

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(data))
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]

    # Create a balanced dataset
    seizure_indices = np.where(labels == 1)[0]
    nonseizure_indices = np.where(labels == 0)[0]

    n_samples = min(len(seizure_indices), len(nonseizure_indices))
    if n_samples == 0:
        raise ValueError("No valid samples found after filtering. Check your data and threshold.")

    print(f"Using {n_samples} samples from each class for balanced dataset")

    seizure_indices = np.random.choice(seizure_indices, n_samples, replace=False)
    nonseizure_indices = np.random.choice(nonseizure_indices, n_samples, replace=False)

    data = np.concatenate((data[seizure_indices], data[nonseizure_indices]), axis=0)
    labels = np.concatenate((labels[seizure_indices], labels[nonseizure_indices]), axis=0)

    # Use stratified sampling for train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_percentage, random_state=0)
    for train_index, val_index in sss.split(data, labels):
        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

    # Create the data loaders
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f"Final dataset shapes - Training: {train_data.shape}, Validation: {val_data.shape}")

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
    PAT_NO = 65

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