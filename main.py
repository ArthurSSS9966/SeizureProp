from steps import init_examination, preprocessing
import mne
import os
import pickle
from datasetConstruct import EDFData


DATA_FOLDER = "data"
RESULT_FOLDER = "result"
PAT_NO = 65
SEIZURE_NO = 0


if __name__ == "__main__":
    # Load the data
    datafile = os.path.join(DATA_FOLDER, f"P{PAT_NO}", f"seizure_{SEIZURE_NO}.pkl")
    # dataset = mne.io.read_raw_edf(datafile, preload=True)

    dataset = pickle.load(open(datafile, "rb"))

    # add the field of seizure number into the edf file
    dataset.seizureNumber = SEIZURE_NO

    # init_examination(dataset, 20, RESULT_FOLDER, start_time=-3, end_time=7)

    dataset = preprocessing(dataset, DATA_FOLDER)

