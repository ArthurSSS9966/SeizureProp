import pickle
import os
from datasetConstruct import EDFData
from steps import init_examination, preprocessing, extract_sEEG_features
from time import time

DATA_FOLDER = "data"
RESULT_FOLDER = "result"
MODEL_FOLDER = "model"
PAT_NOs = [65,66]
preprocessDataset = True

if __name__ == "__main__":

    for PAT_NO in PAT_NOs:

        if preprocessDataset:
            # Load the data
            datafiles = os.path.join(DATA_FOLDER, f"P{PAT_NO}")
            # dataset = mne.io.read_raw_edf(datafile, preload=True)

            for datafile in os.listdir(datafiles):
                # Track the time

                datafile = os.path.join(datafiles, datafile)
                if (datafile.endswith(".pkl") and not datafile.endswith("_CLEANED.pkl")) and not datafile.endswith("_combined.pkl"):
                    start = time()
                    dataset = pickle.load(open(datafile, "rb"))

                    seizure_number = datafile.split("_")[-1].split(".")[0]
                    print(f"Seizure Number: {seizure_number}")
                    dataset.seizure_number = seizure_number
                    # init_examination(dataset, 20, RESULT_FOLDER, start_time=-3, end_time=7)

                    try:
                        dataset = preprocessing(dataset, os.path.join(DATA_FOLDER, f"P{PAT_NO}"))
                    except Exception as e:
                        print(f"Error Preprocessing: {e}")
                        continue

                    print(f"Preprocessing for {datafile} took {time() - start} seconds")

    # results, models = setup_and_train_models(
    #     data_folder="data",
    #     model_folder="checkpoints",
    #     model_names=['CNN1D', 'Wavenet'],  # Only use CNN1D and Wavenet
    #     train=False,
    #     params={'epochs': 40, 'batch_size': 4096}
    # )
    #
    # model_name = 'CNN1D'
    # model = models[model_name]
    #
    # params = {
    #     'threshold': 0.8,
    #     'smooth_window': 50,
    #     'n_seconds': 80,
    #     'seizure_start': 10,
    # }
    #
    # # Run analysis
    # results_propagation = analyze_seizure_propagation(
    #     patient_no=66,
    #     seizure_no=1,
    #     model=model,
    #     data_folder='data',
    #     params=params,
    #     save_results_ind=True
    # )


