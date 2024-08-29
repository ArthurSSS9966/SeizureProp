from utils import notch_filter, butter_bandpass_filter
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import welch
import os


DATA_FOLDER = "data"
PAT_NO = 65


if __name__ == "__main__":
    # Load the data
    datafile = os.path.join(DATA_FOLDER, f"P{PAT_NO}", "seizure_1.pkl")
    dataset = pickle.load(open(datafile, "rb"))

    eegData = dataset["eegData"]
    seizureNumber = dataset["seizureNumber"]
    samplingRate = dataset["samplingRate"]*2
    timeIndex = dataset["normalized_times"]

    # Get time index when normalized time is 0
    timeStartIndex = np.where(timeIndex == 0)[0][0]
    timeStartIndex = int(timeStartIndex - 3 * samplingRate)
    timeEndIndex = int(timeStartIndex + 7 * samplingRate)

    channelNumber = 10

    # Get the first channel from 10 to 15 seconds
    channel1 = eegData[timeStartIndex:timeEndIndex, channelNumber]

    timerange = np.linspace(-4, 6, len(channel1))

    plt.plot(timerange,channel1)
    plt.vlines(0, -1000, 1000, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel 1 Raw")
    plt.legend()
    plt.savefig(f"seizure_{seizureNumber}_raw.png")
    plt.show()

    # Apply notch filter
    channel1_filtered = notch_filter(channel1, fs=dataset['samplingRate'], freq=60)


    # Apply bandpass filter
    channel1_filtered = butter_bandpass_filter(channel1_filtered, lowcut=0.5, highcut=70, fs=dataset['samplingRate'])

    plt.plot(timerange,channel1_filtered)
    plt.vlines(0, -1000, 1000, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Channel 1 Filtered")
    plt.legend()
    plt.savefig(f"seizure_{seizureNumber}_filtered.png")
    plt.show()

    # Bipolar montage
    channel2 = eegData[timeStartIndex:timeEndIndex, channelNumber + 1]
    bipolar = channel1 - channel2

    plt.plot(timerange,bipolar)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Bipolar")
    plt.legend()
    plt.savefig(f"seizure_{seizureNumber}_bipolar.png")
    plt.show()

    # Filter the bipolar montage
    bipolar_filtered = notch_filter(bipolar, fs=dataset['samplingRate'], freq=60)

    bipolar_filtered = butter_bandpass_filter(bipolar_filtered, lowcut=0.5, highcut=70, fs=dataset['samplingRate'])

    plt.plot(timerange, bipolar_filtered)
    plt.vlines(0, -200, 200, color='r', linestyles='dashed', label='Seizure Start')
    plt.title(f"Seizure {seizureNumber}_Bipolar Filtered")
    plt.legend()
    plt.savefig(f"seizure_{seizureNumber}_bipolar_filtered.png")
    plt.show()


    # Calculate the power spectral density
    nperseg = samplingRate//2
    noverlap = nperseg//4

    f, Pxx = welch(channel1, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.title(f"Seizure {seizureNumber}_Channel 1 Power Spectral Density")
    plt.xlim(0, 100)
    plt.savefig(f"seizure_{seizureNumber}_psd.png")
    plt.show()

    f, Pxx = welch(channel1_filtered, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_Channel 1 Filtered Power Spectral Density")
    plt.savefig(f"seizure_{seizureNumber}_filtered_psd.png")
    plt.show()

    f, Pxx = welch(bipolar_filtered, fs=dataset['samplingRate'], nperseg=nperseg, noverlap=noverlap)
    plt.plot(f, Pxx)
    plt.xlim(0, 100)
    plt.title(f"Seizure {seizureNumber}_Bipolar Filtered Power Spectral Density")
    plt.savefig(f"seizure_{seizureNumber}_bipolar_filtered_psd.png")
    plt.show()



