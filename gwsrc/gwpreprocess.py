import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, decimate
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')


class SignalProcessor:
    """Class for signal processing operations adapted for MAT files."""

    @staticmethod
    def butter_bandpass_filter(data, lowcut=1, highcut=255, fs=2000, order=4):
        """Apply bandpass filter to data."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    @staticmethod
    def apply_bipolar_reference(data: np.ndarray, matter_data: pd.DataFrame) -> np.ndarray:
        """Apply bipolar referencing based on electrode spatial arrangement."""
        bipolar_data = np.zeros_like(data)

        # Simple adjacent electrode bipolar referencing
        # You can customize this based on your electrode grid layout
        for i in range(data.shape[1] - 1):
            bipolar_data[:, i] = data[:, i] - data[:, i + 1]

        # Keep the last electrode as reference
        bipolar_data[:, -1] = data[:, -1]

        return bipolar_data

    @staticmethod
    def remove_line_noise(data: np.ndarray, fs: int, line_freqs: list = [60, 120]) -> np.ndarray:
        """Remove line noise using notch filters."""
        from scipy.signal import iirnotch, filtfilt

        cleaned_data = data.copy()

        for freq in line_freqs:
            # Design notch filter
            Q = 30  # Quality factor
            b, a = iirnotch(freq, Q, fs)

            # Apply to all channels
            for i in tqdm(range(cleaned_data.shape[1]), desc=f"Removing {freq}Hz noise"):
                cleaned_data[:, i] = filtfilt(b, a, cleaned_data[:, i])

        return cleaned_data

    @staticmethod
    def apply_whitening(data: np.ndarray, lags: int = 1) -> np.ndarray:
        """Apply whitening using auto-regressive model."""
        data = np.asarray(data)
        whitened = np.zeros_like(data)

        for i in tqdm(range(data.shape[1]), desc="Whitening signals"):
            signal = data[:, i]
            try:
                model = AutoReg(signal, lags=lags).fit()
                predictions = model.predict(start=lags, end=len(signal) - 1)
                whitened[lags:, i] = signal[lags:] - predictions
                whitened[:lags, i] = signal[:lags] - np.mean(signal)
            except:
                # If whitening fails, keep original signal
                whitened[:, i] = signal

        return whitened

    @staticmethod
    def normalize_signal(signal: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """Normalize signal using RobustScaler."""
        if reference is None:
            reference = signal

        normalized = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            scaler = RobustScaler()
            scaler.fit(reference[:, i].reshape(-1, 1))
            normalized[:, i] = scaler.transform(
                signal[:, i].reshape(-1, 1)
            ).squeeze()
        return normalized


def load_mat_and_matter(mat_file, matter_file):
    """Load MAT file and matter CSV, align electrode counts."""

    # Load MAT file
    mat_data = loadmat(mat_file)
    neural_data = mat_data['neural_data']
    sampling_rate = float(mat_data['neural_data_fs'].flatten()[0])

    # Load matter CSV
    matter_data = pd.read_csv(matter_file)
    n_electrodes = len(matter_data)

    print(f"Original neural data shape: {neural_data.shape}")
    print(f"Matter electrodes: {n_electrodes}")

    # Truncate neural data to match matter electrodes (remove dummy electrodes)
    neural_data = neural_data[:, :n_electrodes]
    print(f"Truncated neural data shape: {neural_data.shape}")

    return neural_data, sampling_rate, matter_data


def preprocess_neural_data(neural_data, sampling_rate, matter_data,
                           apply_bipolar=True, apply_line_noise_removal=True,
                           apply_whitening=True, apply_normalization=True):
    """
    Preprocess neural data following the 6-step pipeline:
    1. Scale, 2. Bipolar, 3. Remove line noise, 4. Bandpass filter,
    5. Downsample to 512, 6. Whitening and normalization

    Parameters:
    - neural_data: numpy array of shape (time_points, n_electrodes)
    - sampling_rate: sampling rate in Hz
    - matter_data: electrode information DataFrame
    - apply_bipolar: whether to apply bipolar referencing
    - apply_line_noise_removal: whether to remove line noise
    - apply_whitening: whether to apply whitening
    - apply_normalization: whether to apply normalization
    """

    processor = SignalProcessor()
    processed_data = neural_data.copy().astype(np.float64)

    print(f"Starting 6-step preprocessing pipeline...")
    print(f"Input shape: {processed_data.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")

    # Step 1: Scale data (smart rescaling to reasonable magnitude)
    print("Step 1: Smart rescaling to reasonable magnitude...")
    original_mean = np.mean(np.abs(processed_data))

    # Determine appropriate scaling factor to get mean magnitude in range 0.1-10
    if original_mean == 0:
        scale_factor = 1
        print("  Warning: Zero mean detected, no scaling applied")
    else:
        # Calculate power of 10 to bring mean to ~1
        log_mean = np.log10(original_mean)
        target_power = -np.round(log_mean)
        scale_factor = 10 ** target_power

    processed_data = processed_data * scale_factor
    new_mean = np.mean(np.abs(processed_data))

    print(f"  Original mean magnitude: {original_mean:.6f}")
    print(f"  Scale factor: {scale_factor}")
    print(f"  New mean magnitude: {new_mean:.6f}")
    print(f"  Data range after scaling: {np.min(processed_data):.3f} to {np.max(processed_data):.3f}")

    # Step 2: Apply bipolar referencing
    if apply_bipolar:
        print("Step 2: Applying bipolar referencing...")
        processed_data = processor.apply_bipolar_reference(processed_data, matter_data)
    else:
        print("Step 2: Skipping bipolar referencing...")

    # Step 3: Remove line noise
    if apply_line_noise_removal:
        print("Step 3: Removing line noise (60Hz, 120Hz)...")
        processed_data = processor.remove_line_noise(
            processed_data, sampling_rate, line_freqs=[60, 120]
        )
    else:
        print("Step 3: Skipping line noise removal...")

    # Step 4: Apply bandpass filter
    print("Step 4: Applying bandpass filter (1-255 Hz)...")
    for i in tqdm(range(processed_data.shape[1]), desc="Filtering channels"):
        processed_data[:, i] = processor.butter_bandpass_filter(
            processed_data[:, i], lowcut=1, highcut=255, fs=sampling_rate
        )

    # Step 5: Downsample to 512 Hz
    print("Step 5: Downsampling to 512 Hz...")
    try:
        if sampling_rate > 512:
            factor = int(sampling_rate // 512)
            processed_data = decimate(processed_data, factor, axis=0)
            new_sampling_rate = sampling_rate / factor
            print(f"  Downsampled from {sampling_rate} Hz to {new_sampling_rate} Hz")
            print(f"  New shape: {processed_data.shape}")
        else:
            new_sampling_rate = sampling_rate
            print(f"  No downsampling needed, keeping {sampling_rate} Hz")
    except Exception as e:
        print(f"  Downsampling failed: {e}")
        new_sampling_rate = sampling_rate

    # Step 6: Apply whitening and normalization
    print("Step 6: Applying whitening and normalization...")

    # First whitening
    if apply_whitening:
        try:
            print("  Applying whitening...")
            processed_data = processor.apply_whitening(processed_data)
        except Exception as e:
            print(f"  Whitening failed: {e}")

    # Then normalization
    if apply_normalization:
        try:
            print("  Applying normalization...")
            processed_data = processor.normalize_signal(processed_data)
        except Exception as e:
            print(f"  Normalization failed: {e}")

    print(f"Final processed shape: {processed_data.shape}")
    print(f"Final data range: {np.min(processed_data):.3f} to {np.max(processed_data):.3f}")

    return processed_data, new_sampling_rate


def process_single_file(mat_file, matter_file, output_folder, apply_bipolar=False):
    """Process a single MAT file."""

    print(f"\nProcessing: {os.path.basename(mat_file)}")
    print("=" * 50)

    try:
        # Load data
        neural_data, sampling_rate, matter_data = load_mat_and_matter(mat_file, matter_file)

        # Preprocess
        processed_data, new_sampling_rate = preprocess_neural_data(
            neural_data, sampling_rate, matter_data,
            apply_bipolar=apply_bipolar,
            apply_line_noise_removal=True,
            apply_whitening=False,
            apply_normalization=True
        )

        # Save processed data as pickle file
        output_filename = os.path.basename(mat_file).replace('.mat', '_processed.pkl')
        output_path = os.path.join(output_folder, output_filename)

        # Prepare data for saving
        output_data = {
            'neural_data_processed': processed_data,
            'neural_data_fs': new_sampling_rate,
            'matter_data': matter_data,  # Keep as DataFrame
            'original_shape': neural_data.shape,
            'processed_shape': processed_data.shape,
            'processing_info': {
                'step1_scaling': f'Smart rescaling with factor',
                'step2_bipolar_referencing': apply_bipolar,
                'step3_line_noise_removal': 'Notch filters at 60Hz, 120Hz',
                'step4_bandpass_filter': '1-255 Hz',
                'step5_downsampled': new_sampling_rate != sampling_rate,
                'step6_whitening': False,
                'step6_normalization': False
            }
        }

        # Save as pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"✓ Saved: {output_path}")

        return True, output_path

    except Exception as e:
        print(f"✗ Error processing {mat_file}: {e}")
        return False, None


def process_patient_files(patient_id, baseline_folder, csv_folder, output_folder):
    """Process all MAT files for a specific patient and save as one pkl file."""

    print(f"\nProcessing Patient {patient_id}")
    print("=" * 60)

    # Find MAT files for this patient
    mat_pattern = os.path.join(baseline_folder, f"{patient_id}_baseline_*.mat")
    mat_files = glob.glob(mat_pattern)
    mat_files.sort()

    # Find matter CSV file
    matter_file = os.path.join(csv_folder, f"{patient_id}_matter.csv")

    if not os.path.exists(matter_file):
        print(f"Matter file not found: {matter_file}")
        return False

    if not mat_files:
        print(f"No MAT files found for {patient_id}")
        return False

    print(f"Found {len(mat_files)} MAT files for {patient_id}")

    # Load matter data once
    matter_data = pd.read_csv(matter_file)
    print(f"Matter data: {len(matter_data)} electrodes")

    # Process all MAT files for this patient
    patient_data = {
        'patient_id': patient_id,
        'matter_data': matter_data,
        'recordings': [],
        'processing_summary': {
            'total_files': len(mat_files),
            'successful_files': 0,
            'failed_files': 0,
            'total_duration_seconds': 0
        }
    }

    for i, mat_file in enumerate(mat_files):
        print(f"\n  Processing file {i + 1}/{len(mat_files)}: {os.path.basename(mat_file)}")

        try:
            # Load MAT file
            neural_data, sampling_rate, _ = load_mat_and_matter(mat_file, matter_file)

            # Preprocess
            processed_data, new_sampling_rate = preprocess_neural_data(
                neural_data, sampling_rate, matter_data,
                apply_bipolar=True,
                apply_line_noise_removal=True,
                apply_whitening=True,
                apply_normalization=True
            )

            # Calculate duration
            duration_seconds = processed_data.shape[0] / new_sampling_rate

            # Store this recording
            recording_data = {
                'file_number': i + 1,
                'original_filename': os.path.basename(mat_file),
                'neural_data_processed': processed_data,
                'sampling_rate': new_sampling_rate,
                'original_shape': neural_data.shape,
                'processed_shape': processed_data.shape,
                'duration_seconds': duration_seconds,
                'processing_info': {
                    'step1_scaling': 'Smart rescaling applied',
                    'step2_bipolar_referencing': True,
                    'step3_line_noise_removal': 'Notch filters at 60Hz, 120Hz',
                    'step4_bandpass_filter': '1-255 Hz',
                    'step5_downsampled': new_sampling_rate != sampling_rate,
                    'step6_whitening': True,
                    'step6_normalization': True
                }
            }

            patient_data['recordings'].append(recording_data)
            patient_data['processing_summary']['successful_files'] += 1
            patient_data['processing_summary']['total_duration_seconds'] += duration_seconds

            print(f"    ✓ Processed: {processed_data.shape} @ {new_sampling_rate} Hz, {duration_seconds:.1f}s")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            patient_data['processing_summary']['failed_files'] += 1

    # Save patient data as single pkl file
    output_filename = f"{patient_id}_processed.pkl"
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'wb') as f:
        pickle.dump(patient_data, f)

    # Summary
    summary = patient_data['processing_summary']
    total_duration_min = summary['total_duration_seconds'] / 60

    print(f"\n✓ Patient {patient_id} completed!")
    print(f"  Saved to: {output_filename}")
    print(f"  Successful files: {summary['successful_files']}/{summary['total_files']}")
    print(f"  Total duration: {total_duration_min:.1f} minutes")

    return True


def main():
    """Main processing function."""

    # Set paths
    baseline_folder = r"D:\Blcdata\Baseline"
    csv_folder = r"D:\Blcdata\Electrodes"
    output_folder = r"D:\BlcRepo\LabCode\SeizureProp\data\gwbaseline_alt"

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print("Neural Data Preprocessing Pipeline")
    print("=" * 60)

    # Get available patients
    mat_files = glob.glob(os.path.join(baseline_folder, "P*_baseline_*.mat"))
    patients = set()
    for mat_file in mat_files:
        filename = os.path.basename(mat_file)
        match = re.match(r'(P\d+)_baseline_\d+\.mat', filename)
        if match:
            patients.add(match.group(1))

    patients = sorted(list(patients))
    print(f"Found patients: {patients}")

    # Choose processing mode
    print("\nProcessing options:")
    print("1. Process all patients")
    print("2. Process specific patient")
    print("3. Process single file")

    choice = input("Choose option (1-3): ").strip()

    if choice == "1":
        # Process all patients
        successful_patients = 0
        for patient_id in patients:
            success = process_patient_files(patient_id, baseline_folder, csv_folder, output_folder)
            if success:
                successful_patients += 1

        print(f"\nProcessing complete!")
        print(f"Successful patients: {successful_patients}/{len(patients)}")

    elif choice == "2":
        # Process specific patient
        print(f"Available patients: {patients}")
        patient_id = input("Enter patient ID (e.g., P061): ").strip()
        if patient_id in patients:
            success = process_patient_files(patient_id, baseline_folder, csv_folder, output_folder)
            if success:
                print(f"✓ Successfully processed {patient_id}")
            else:
                print(f"✗ Failed to process {patient_id}")
        else:
            print(f"Patient {patient_id} not found")

    elif choice == "3":
        # Process single patient from file path
        mat_file = input("Enter full path to any MAT file for the patient: ").strip()
        patient_match = re.search(r'(P\d+)_baseline', os.path.basename(mat_file))
        if patient_match:
            patient_id = patient_match.group(1)

            if patient_id in patients:
                success = process_patient_files(patient_id, baseline_folder, csv_folder, output_folder)
                if success:
                    print(f"✓ Successfully processed {patient_id}")
                else:
                    print(f"✗ Failed to process {patient_id}")
            else:
                print(f"Patient {patient_id} not found in available patients")
        else:
            print("Invalid file name format")


if __name__ == "__main__":
    import re

    main()