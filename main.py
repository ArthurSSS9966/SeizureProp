from steps import init_examination
import mne
import pickle
import os


DATA_FOLDER = "data"
RESULT_FOLDER = "result"
PAT_NO = 65
SEIZURE_NO = 1


if __name__ == "__main__":
    # Load the data
    datafile = os.path.join(DATA_FOLDER, f"P{PAT_NO}", f"SZ{SEIZURE_NO}.EDF")
    dataset = mne.io.read_raw_edf(datafile, preload=True)

    # add the field of seizure number into the edf file
    dataset.seizureNumber = SEIZURE_NO

    init_examination(dataset, 20, RESULT_FOLDER, start_time=-3, end_time=7)

