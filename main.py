import os
import pickle
import torch

from datasetConstruct import EDFData
from steps import init_examination, preprocessing
from datasetConstruct import load_seizure, create_dataset
from models import train_using_optimizer, evaluate_model

DATA_FOLDER = "data"
RESULT_FOLDER = "result"
MODEL_FOLDER = "model"
PAT_NO = 66


def train_test(model, PAT_NO, data_folder, TRAIN=True, epochs=10):
    # Load the data
    data_folder = os.path.join(data_folder, f"P{PAT_NO}")
    seizure = load_seizure(data_folder)

    # Create the dataset
    train_loader, val_loader = create_dataset(seizure, train_percentage=0.2)

    _, channels, time_steps = train_loader.dataset[0][0].shape

    # Create the model
    model1 = model(input_dim=channels, kernel_size=time_steps, output_dim=2, lr=0.001, dropout=0.2)

    if TRAIN:

        # Train the model
        train_loss, val_los, val_accuracy = train_using_optimizer(model1, train_loader,
                                                                  val_loader, MODEL_FOLDER,
                                                                  epochs=epochs)

    else:
        model1.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, f"{model1.__class__.__name__}_epoch{epochs}.pth")))

    val_loss, val_accuracy = evaluate_model(model1, val_loader)

    return train_loss, val_loss, val_accuracy

# def validate_on_seizure(model, PAT_NO, seizure_NO):


if __name__ == "__main__":
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
