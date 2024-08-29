import os
import numpy as np
import matplotlib.pyplot as plt
from utils import notch_filter, butter_bandpass_filter
from scipy.signal import welch

def init_examination(dataset, channelNumber, RESULT_FOLDER, start_time=-3, end_time=7):
    '''
    Function to initialize the examination of the seizure data
    :param dataset:
    :param channelNumber:
    :param RESULT_FOLDER:
    :param start_time:
    :param end_time:
    :return:
    '''
    eegData = dataset["eegData"]
    samplingRate = dataset["samplingRate"] * 2
    timeIndex = dataset["normalized_times"]
    seizureNumber = dataset["seizureNumber"]

    # Get time index when normalized time is 0
    timeStartIndex = np.where(timeIndex == 0)[0][0]
    timeStartIndex = int(timeStartIndex + start_time * samplingRate)
    timeEndIndex = int(timeStartIndex + end_time * samplingRate)

    # Get the first channel from 10 to 15 seconds
    channel1 = eegData[timeStartIndex:timeEndIndex, channelNumber]

    timerange = np.linspace(start_time - 1, end_time - 1, len(channel1))

    plt.plot(timerange, channel1)
    plt.vlines(0, -1000, 1000, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel 1 Raw")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_raw.svg")
    plt.savefig(save_location)
    plt.show()

    # Apply notch filter
    channel1_filtered = notch_filter(channel1, fs=dataset['samplingRate'], freq=60)

    # Apply bandpass filter
    channel1_filtered = butter_bandpass_filter(channel1_filtered, lowcut=0.5, highcut=70, fs=dataset['samplingRate'])

    plt.plot(timerange, channel1_filtered)
    plt.vlines(0, -1000, 1000, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel 1 Filtered")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_filtered.svg")
    plt.savefig(save_location)
    plt.show()

    # Bipolar montage
    channel2 = eegData[timeStartIndex:timeEndIndex, channelNumber + 1]
    bipolar = channel1 - channel2

    plt.plot(timerange, bipolar)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Bipolar")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar.svg")
    plt.savefig(save_location)
    plt.show()

    # Filter the bipolar montage
    bipolar_filtered = notch_filter(bipolar, fs=dataset['samplingRate'], freq=60)

    bipolar_filtered = butter_bandpass_filter(bipolar_filtered, lowcut=0.5, highcut=70, fs=dataset['samplingRate'])

    plt.plot(timerange, bipolar_filtered)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Bipolar Filtered")
    plt.legend()
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar_filtered.svg")
    plt.savefig(save_location)
    plt.show()

    # Calculate the power spectral density
    nperseg = samplingRate // 2
    noverlap = nperseg // 4

    f, Pxx = welch(channel1, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.title(f"Seizure {seizureNumber}_Channel 1 Power Spectral Density")
    plt.xlim(0, 100)
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_psd.png")
    plt.savefig(save_location)
    plt.show()

    f, Pxx = welch(channel1_filtered, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_Channel 1 Filtered Power Spectral Density")
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_filtered_psd.png")
    plt.savefig(save_location)
    plt.show()

    f, Pxx = welch(bipolar_filtered, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_Bipolar Filtered Power Spectral Density")
    save_location = os.path.join(RESULT_FOLDER, f"seizure_{seizureNumber}_bipolar_filtered_psd.png")
    plt.savefig(save_location)
    plt.show()