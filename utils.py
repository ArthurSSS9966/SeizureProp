from scipy.signal import butter, filtfilt, sosfilt
from scipy.signal import iirnotch


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
