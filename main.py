import os
import pickle

from datasetConstruct import EDFData
from steps import init_examination, preprocessing

DATA_FOLDER = "data"
RESULT_FOLDER = "result"
PAT_NO = 66


if __name__ == "__main__":
    # Load the data
    datafiles = os.path.join(DATA_FOLDER, f"P{PAT_NO}")
    # dataset = mne.io.read_raw_edf(datafile, preload=True)

    for datafile in os.listdir(datafiles):
        datafile = os.path.join(datafiles, datafile)
        if datafile.endswith(".pkl"):
            dataset = pickle.load(open(datafile, "rb"))

            # init_examination(dataset, 20, RESULT_FOLDER, start_time=-3, end_time=7)

            try:
                dataset = preprocessing(dataset, os.path.join(DATA_FOLDER, f"P{PAT_NO}"))
            except Exception as e:
                print(f"Error: {e}")
                continue
