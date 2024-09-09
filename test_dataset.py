'''
This file is just to test the Kaggle dataset, how it looks and how to use it to test the model
'''

import pandas as pd
import numpy as np
import os
from datasetConstruct import CustomDataset
from torch.utils.data import DataLoader
from models import CNN1D, train_using_optimizer

DATA_FOLDER = "D:/Blcdata/seizure/Test_data"

if __name__ == "__main__":
    # Load the .npz file
    Train_data = np.load(os.path.join(DATA_FOLDER, "eeg-seizure_train.npz"))
    Valid_data = np.load(os.path.join(DATA_FOLDER, "eeg-seizure_val.npz"))

    # Load the data
    X_train = Train_data.f.train_signals
    Y_train = Train_data.f.train_labels

    X_valid = Valid_data.f.val_signals
    Y_valid = Valid_data.f.val_labels

    # Repeat the labels with the same order as the data
    Y_train = np.array([Y_train for _ in range(X_train.shape[1])]).flatten()
    Y_valid = np.array([Y_valid for _ in range(X_valid.shape[1])]).flatten()

    # Reshape the dataset into single channel
    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1, -1)
    X_valid = X_valid.reshape(X_valid.shape[0]*X_valid.shape[1], 1, -1)

    time_steps = X_train.shape[2]
    channels = X_train.shape[1]

    # Create the dataset
    dataset = CustomDataset(X_train, Y_train)
    valdataset = CustomDataset(X_valid, Y_valid)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    valdataloader = DataLoader(valdataset, batch_size=32, shuffle=True)

    # Create the model
    model = CNN1D(input_dim=channels, kernel_size=time_steps, output_dim=2)

    # Train the model
    train_using_optimizer(model, dataloader, valdataloader)