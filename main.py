from steps import init_examination
import pickle
import os


DATA_FOLDER = "data"
RESULT_FOLDER = "result"
PAT_NO = 65


if __name__ == "__main__":
    # Load the data
    datafile = os.path.join(DATA_FOLDER, f"P{PAT_NO}", "seizure_1.pkl")
    dataset = pickle.load(open(datafile, "rb"))

    init_examination(dataset, 0, RESULT_FOLDER, start_time=-3, end_time=7)