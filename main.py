import pickle
import os

from steps import init_examination, preprocessing, setup_and_train_models, analyze_seizure_propagation

DATA_FOLDER = "data"
RESULT_FOLDER = "result"
MODEL_FOLDER = "model"
PAT_NO = 65
preprocessDataset = False

if __name__ == "__main__":

    if preprocessDataset:
        # Load the data
        datafiles = os.path.join(DATA_FOLDER, f"P{PAT_NO}")
        # dataset = mne.io.read_raw_edf(datafile, preload=True)

        for datafile in os.listdir(datafiles):
            datafile = os.path.join(datafiles, datafile)
            if (datafile.endswith(".pkl") and not datafile.endswith("_CLEANED.pkl")):
                dataset = pickle.load(open(datafile, "rb"))

                # init_examination(dataset, 20, RESULT_FOLDER, start_time=-3, end_time=7)

                try:
                    dataset = preprocessing(dataset, os.path.join(DATA_FOLDER, f"P{PAT_NO}"))
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    results, models = setup_and_train_models(
        data_folder="data",
        model_folder="checkpoints",
        model_names=['CNN1D', 'Wavenet'],  # Only use CNN1D and Wavenet
        train=False,
        params={'epochs': 40, 'batch_size': 4096}
    )

    model_name = 'CNN1D'
    model = models[model_name]

    params = {
        'threshold': 0.8,
        'smooth_window': 50,
        'n_seconds': 80,
        'seizure_start': 10,
    }

    # Run analysis
    results_propagation = analyze_seizure_propagation(
        patient_no=66,
        seizure_no=1,
        model=model,
        data_folder='data',
        params=params,
        save_results_ind=True
    )


