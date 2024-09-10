import os
import numpy as np
import matplotlib.pyplot as plt
from utils import notch_filter, butter_bandpass_filter
from scipy.signal import welch, decimate
from meegkit import dss
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.ar_model import AutoReg


def init_examination(dataset, channelNumber, RESULT_FOLDER, start_time=-3, end_time=7):
    '''
    Function to initialize the examination of the seizure data,
    This will plot the raw data, filtered data, bipolar montage, filtered bipolar montage,
    and power spectral density of the raw data, filtered data, and filtered bipolar montage
    :param dataset: sEEG .edf data
    :param channelNumber: The channel number to be examined
    :param RESULT_FOLDER: The folder to save the results
    :param start_time: The start time of the examination window
    :param end_time: The end time of the examination window
    :return:
    '''
    eegData, timerange = dataset[:]
    eegData = eegData.T * 1e6
    samplingRate = int(dataset.info["sfreq"])
    timeIndex = dataset.annotations.onset
    seizureNumber = dataset.seizureNumber
    channelName = dataset.ch_names[channelNumber]

    print(f"Seizure {seizureNumber} Channel {channelName} is being examined...")

    # Find the index of timeIndex where it matches 'sz'
    timeStartIndex = np.where(dataset.annotations.description == 'SZ')[0][0]
    timeStartIndex = int(timeIndex[timeStartIndex] * samplingRate)
    timeStartIndex = int(timeStartIndex + start_time * samplingRate)
    timeEndIndex = int(timeStartIndex + end_time * samplingRate)

    # Get the first channel from 10 to 15 seconds
    channel1 = eegData[timeStartIndex:timeEndIndex, channelNumber]

    timerange = np.linspace(start_time - 1, end_time - 1, len(channel1))

    plt.plot(timerange, channel1)
    plt.vlines(0, -400, 400, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel {channelName} Raw")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"{channelName}seizure_{seizureNumber}_raw.svg")
    plt.savefig(save_location)
    plt.show()

    # Apply notch filter
    channel1_filtered = notch_filter(channel1, fs=samplingRate, freq=60)

    # Apply bandpass filter
    channel1_filtered = butter_bandpass_filter(channel1_filtered, lowcut=0.5, highcut=70, fs=samplingRate)

    plt.plot(timerange, channel1_filtered)
    plt.vlines(0, -400, 400, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel {channelName} Filtered")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_filtered.svg")
    plt.savefig(save_location)
    plt.show()

    # Bipolar montage
    channel2 = eegData[timeStartIndex:timeEndIndex, channelNumber + 1]
    bipolar = channel1 - channel2

    plt.plot(timerange, bipolar)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_{channelName}_Bipolar")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar.svg")
    plt.savefig(save_location)
    plt.show()

    # Filter the bipolar montage
    bipolar_filtered = notch_filter(bipolar, fs=samplingRate, freq=60)

    bipolar_filtered = butter_bandpass_filter(bipolar_filtered, lowcut=0.5, highcut=70, fs=samplingRate)

    plt.plot(timerange, bipolar_filtered)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_{channelName}_Bipolar Filtered")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar_filtered.svg")
    plt.savefig(save_location)
    plt.show()

    # Calculate the power spectral density
    nperseg = samplingRate // 2
    noverlap = nperseg // 4

    f, Pxx = welch(channel1, fs=samplingRate, nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.title(f"Seizure {seizureNumber}_Channel {channelName} Power Spectral Density")
    plt.xlim(0, 100)
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_psd.png")
    plt.savefig(save_location)
    plt.show()

    f, Pxx = welch(channel1_filtered, fs=samplingRate, nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_Channel {channelName} Filtered Power Spectral Density")
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_filtered_psd.png")
    plt.savefig(save_location)
    plt.show()

    f, Pxx = welch(bipolar_filtered, fs=samplingRate, nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_{channelName}_Bipolar Filtered Power Spectral Density")
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar_filtered_psd.png")
    plt.savefig(save_location)
    plt.show()


def preprocessing(dataset, DATA_FOLDER):
    eegData, timerange = dataset.ictal
    referenceData, timerange_interictal = dataset.interictal

    eegData = eegData.T * 1e6
    referenceData = referenceData.T * 1e6
    samplingRate = int(dataset.info["sfreq"])

    # Remove line noise from each electrode
    cleanedData = remove_line_each_electrode(eegData, dataset)
    cleanedReferenceData = remove_line_each_electrode(referenceData, dataset)

    # Apply the bandpass filter
    for i in range(cleanedData.shape[1]):
        cleanedData[:, i] = butter_bandpass_filter(cleanedData[:, i], lowcut=1, highcut=127, fs=samplingRate)
        cleanedReferenceData[:, i] = butter_bandpass_filter(cleanedReferenceData[:, i], lowcut=1, highcut=127, fs=samplingRate)

    # Downsample the data to 128 Hz
    cleanedData = decimate(cleanedData, samplingRate//128, axis=0)
    cleanedReferenceData = decimate(cleanedReferenceData, samplingRate//128, axis=0)

    # Apply whitening using autoregressive model
    cleanedData = apply_whitening(cleanedData, lags=1)
    cleanedReferenceData = apply_whitening(cleanedReferenceData, lags=1)

    # Normalize the data
    cleanedData = normalize_signal(cleanedData, cleanedReferenceData)

    # Store cleaned data
    save_location = os.path.join(DATA_FOLDER, f"seizure_{dataset.seizureNumber}_CLEANED.pkl")
    dataset.ictal = cleanedData
    dataset.interictal = cleanedReferenceData
    dataset.save(save_location)

    return dataset


def normalize_signal(signal, reference):
    """
    Normalize the signal using RobustScaler based on interictal data.
    Args:
        signal (numpy array): The input signal to be normalized.
        interictal_data (numpy array): Interictal data used for normalization.
    Returns:
        numpy array: The normalized signal.
    """
    scaler = RobustScaler()
    scaler.fit(reference.reshape(-1, 1))  # Fit scaler on interictal data
    normalized_signal = scaler.transform(signal.reshape(-1, 1)).flatten()
    return normalized_signal


def apply_whitening(signal, lags=1):
    '''
    Apply whitening to the data using auto-regressive model
    :param data: Data
    :param lags: Number of lags
    :return: Whitened data
    '''
    model = AutoReg(signal, lags=lags)
    model_fitted = model.fit()
    prewhitened_signal = signal - model_fitted.predict(start=lags)
    return prewhitened_signal


def remove_line_each_electrode(data, ele):
    '''
    Remove line noise from each electrode
    :param data: Data
    :param ele: Electrode
    :return: Data after removing line noise
    '''
    for e in tqdm(ele.channels['Channel']):
        data[:, e] = remove_line(data[:, e], [60, 100])
    return data


def remove_line(x1, lineF, Fs=2000):
    '''

    :param x1:
    :param lineF:
    :param Fs:
    :return:
    '''
    print("Start Line noise removal")
    xret = np.array(x1)
    for f0 in lineF:
        xret, _ = dss.dss_line_iter(xret, f0, Fs)
    print("Removal Line noise removal Complete")
    return xret


