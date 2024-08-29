from scipy.signal import coherence, welch, csd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from mne_connectivity import spectral_connectivity_epochs

class ConnectivityCalculator:
    def __init__(self, data, fs):
        self.Cxy_total = None
        self.f = None
        self.data = data
        self.fs = fs
        self.time_window = 1  # 1 second
        self.method = None

    def calculate_connectivity(self, signal1, signal2, method):
        f, Cxy = None, None

        if method == 'square_coherence':
            nperseg = self.fs * 1
            noverlap = nperseg // 2
            highest_freq = 200

            # Step 1: Compute cross-spectrum and power spectral densities
            f, Pxx = welch(signal1, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pyy = welch(signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pxy = csd(signal1, signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)

            # Step 2: Compute coherence (Cxy)
            Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)

            # Filter out frequencies above 100 Hz
            f = f[f <= highest_freq]
            Cxy = Cxy[:len(f)]

        elif method == 'lagged_coherence':
            nperseg = self.fs * 1
            noverlap = nperseg // 2
            highest_freq = 200

            # Step 1: Compute cross-spectrum and power spectral densities
            f, Pxx = welch(signal1, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pyy = welch(signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pxy = csd(signal1, signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)

            # Step 2: Compute lagged coherence (LC_xy)
            Cxy = np.imag(Pxy) / np.sqrt(Pxx * Pyy - np.real(Pxy) ** 2)

            # Filter out frequencies above 100 Hz
            f = f[f <= highest_freq]
            Cxy = np.abs(Cxy[:len(f)])

        elif method == 'granger_causality':
            max_lag = 10
            results = grangercausalitytests(np.array([signal1, signal2]).T, max_lag, verbose=False)
            f = np.arange(1, max_lag + 1)
            Cxy = [results[lag][0]['ssr_ftest'][0] for lag in f]

        else:
            raise ValueError('Method not supported')

        return f, Cxy

    def calculate_connectivity_all(self, method):
        n_channels = self.data.shape[0]
        self.method = method
        f, Cxy_total = None, []

        for i in range(n_channels):
            for j in range(n_channels):
                signal1 = self.data[i]
                signal2 = self.data[j]

                f, Cxy = self.calculate_connectivity(signal1, signal2, method)

                Cxy_total.append(Cxy)

        Cxy_total = np.array(Cxy_total)

        self.f = f
        self.Cxy_total = Cxy_total

        return f, Cxy_total

    def plot_connectivity(self):
        if self.Cxy_total is None or self.f is None:
            raise ValueError('No connectivity data calculated')

        n_channels = self.data.shape[0]

        # Calculate the average value for each channel
        Cxy_avg = np.mean(self.Cxy_total, axis=1)
        Cxy_avg = Cxy_avg.reshape(n_channels, n_channels)

        # Plot the coherence using confusion matrix
        plt.figure()
        plt.imshow(Cxy_avg, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.title('Connectivity')
        plt.savefig(f'connectivity_{self.method}.png')
        plt.show()


if __name__ == '__main__':
    fs = 2000
    duration = 2
    t = np.linspace(0, duration, duration*fs)  # 2 seconds
    freq = 100  # 100 Hz

    # 100 Hz + random noise
    signal1 = 5 * np.sin(2 * np.pi * freq * t) + np.random.rand(len(t)) - 0.5
    # 200 Hz + random noise
    signal2 = 5 * np.sin(2 * np.pi * freq * 2 * t) + np.random.rand(len(t)) - 0.5
    # 100 Hz with 0.05s lag + random noise
    signal3 = 5 * np.sin(2 * np.pi * freq * t + 0.05) + np.random.rand(len(t)) - 0.5
    # Random Noise
    signal4 = 5 * np.random.rand(len(t)) - 1

    # Create four channels of data
    signalData = np.array([signal1, signal2, signal3, signal4])

    cc = ConnectivityCalculator(signalData, fs)

    # Calculate lagged Coherence
    f_lagged, Cxy_lagged = cc.calculate_connectivity_all('lagged_coherence')
    cc.plot_connectivity()

    # Calculate Coherence
    f, Cxy = cc.calculate_connectivity_all('square_coherence')
    cc.plot_connectivity()

    # Plot the coherence
    plt.figure()
    plt.plot(f_lagged, Cxy_lagged[1], label='Lagged Coherence')
    plt.plot(f, Cxy[1], label='Coherence')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.ylim([-1, 1])
    plt.grid()
    plt.legend()
    plt.savefig('coherence.png')
    plt.show()

    # Calculate Granger Causality
    f_granger, Cxy_granger = cc.calculate_connectivity_all('granger_causality')
    cc.plot_connectivity()

    # Plot the Granger Causality
    plt.figure()
    plt.plot(f_granger, Cxy_granger[1], label='Granger Causality')
    plt.xlabel('Lag')
    plt.ylabel('F-statistic')
    plt.grid()
    plt.legend()
    plt.savefig('granger_causality.png')
    plt.show()

