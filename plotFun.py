import os
import matplotlib.pyplot as plt
import numpy as np

def plot_time_limited_heatmap(data, time_axis, n_seconds=None, preictal_boundary=None,
                              title="", cmap='cool', clim=None, save_path=None, flip_yaxis=False):
    """
    Plot heatmap with time limitation.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array of data to plot
    time_axis : numpy.ndarray
        Time axis values
    n_seconds : float, optional
        Number of seconds to plot (from start)
    preictal_boundary : int, optional
        Index where preictal period ends
    """
    # Calculate the index corresponding to n_seconds
    if n_seconds is not None:
        time_mask = time_axis <= n_seconds
        plot_data = data[:, time_mask]
        plot_time = time_axis[time_mask]
        if preictal_boundary is not None:
            preictal_boundary = min(preictal_boundary, len(plot_time) - 1)
    else:
        plot_data = data
        plot_time = time_axis

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(plot_time, np.arange(data.shape[0]), plot_data,
                   cmap=cmap, shading='auto')

    if preictal_boundary is not None and preictal_boundary < len(plot_time):
        plt.axvline(x=plot_time[preictal_boundary], color='r', linestyle='--',
                    label='Preictal Boundary')

    # Optimize tick marks
    n_ticks = min(20, len(plot_time))
    tick_indices = np.linspace(0, len(plot_time) - 1, n_ticks, dtype=int)
    plt.xticks(plot_time[tick_indices], np.round(plot_time[tick_indices], 2),
               fontsize=12)
    if flip_yaxis:
        # Flip the y-axis
        plt.gca().invert_yaxis()

    plt.colorbar(label='Amplitude')

    if clim is not None:
        plt.clim(clim[0], clim[1])

    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_eeg_style(data, sampling_rate=None, spacing_factor=1,
                   linewidth=0.5, color='black', figsize=(15, 10)):
    """
    Create an EEG-style plot from 2D data.

    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (n_channels, n_samples)
    sampling_rate : float, optional
        Sampling rate for time axis
    spacing_factor : float
        Factor to control spacing between channels
    linewidth : float
        Width of the EEG lines
    color : str
        Color of the lines
    figsize : tuple
        Figure size (width, height)
    """

    # Get dimensions
    n_channels, n_samples = data.shape

    # Create time axis
    if sampling_rate is not None:
        time = np.arange(n_samples) / sampling_rate
    else:
        time = np.arange(n_samples)

    # Calculate spacing
    # Normalize data (use min/max for each channel) for consistent spacing
    data_normalized = (data - data.min(axis=1)[:, None]) / \
                      (data.max(axis=1) - data.min(axis=1))[:, None] * 4

    # Create offset for each channel
    offsets = np.arange(n_channels) * spacing_factor

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot each channel
    for i in range(n_channels):
        plt.plot(time,
                 data_normalized[i] + offsets[n_channels - 1 - i],  # Reverse order
                 color=color,
                 linewidth=linewidth)

    # Customize the plot
    plt.box(False)  # Remove the box
    plt.yticks(offsets, [f'Ch {n_channels - i}' for i in range(n_channels)])

    if sampling_rate is not None:
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Samples')

    plt.title('EEG')

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    return plt.gcf()
