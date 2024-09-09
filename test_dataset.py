'''
This file is just to test the Kaggle dataset, how it looks and how to use it to test the model
'''

import pandas as pd
import numpy as np
import os
from datasetConstruct import CustomDataset
from torch.utils.data import DataLoader
from models import CNN1D, train_using_optimizer, Wavenet, LSTM

DATA_FOLDER = "D:/Blcdata/seizure/Test_data"
MODEL_FOLDER = "D:/Blcdata/seizure/Model"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

if __name__ == "__main__":
    '''
    A dataset created from EEG signals from the CHB-MIT Scalp EEG Database, specifically from Patient 15.
    
    Using the raw EEG signals, windows of 30 seconds (7680 samples each) were randomly sampled.
    
    Only train and validation sets are included. 
    The training samples are sourced from recordings that were recorded earlier than validation samples.
    
    The {set}_X.npy are files containing the numpy arrays for multichannel signals and have a shape of n x w x c where:
    
        n = number of samples in the set
        w = number of signal sample in a window sample
        c = number of channels
    
    The {set}_y.npy are files containing the target for each window. 
    It is a continuous measure representing the number of seconds from the most recent sample in the window to the next epilepsy event.
    
    This dataset was created to be used in a regression task. 
    However, y can easily be converted to binary or a multivariate variable and be used for classification using specified thresholds.
    '''
    # Load the .npz file
    Train_data = np.load(os.path.join(DATA_FOLDER, "eeg-seizure_train.npz"))
    Valid_data = np.load(os.path.join(DATA_FOLDER, "eeg-seizure_val.npz"))

    # Load the data
    X_train = Train_data.f.train_signals
    Y_train = Train_data.f.train_labels

    X_valid = Valid_data.f.val_signals
    Y_valid = Valid_data.f.val_labels

    # The data is aranged as n*t*c, we need to reshape it to n*c*t
    X_train = X_train.transpose(0, 2, 1)
    X_valid = X_valid.transpose(0, 2, 1)

    # # Repeat the labels with the same order as the data
    # Y_train = np.array([Y_train for _ in range(X_train.shape[1])]).flatten()
    # Y_valid = np.array([Y_valid for _ in range(X_valid.shape[1])]).flatten()
    #
    # # The data is then reshaped to (n*c) * 1 * t
    # X_train = X_train.reshape(-1, 1, X_train.shape[2])
    # X_valid = X_valid.reshape(-1, 1, X_valid.shape[2])

    time_steps = X_train.shape[2]
    channels = X_train.shape[1]

    # Create the dataset
    dataset = CustomDataset(X_train, Y_train)
    valdataset = CustomDataset(X_valid, Y_valid)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    valdataloader = DataLoader(valdataset, batch_size=32, shuffle=True)

    # Create the model
    model1 = CNN1D(input_dim=channels, kernel_size=time_steps, output_dim=2)
    model2 = Wavenet(input_dim=channels, output_dim=2, kernel_size=time_steps)
    model3 = LSTM(input_dim=channels, output_dim=2)

    # Train the model
    train_using_optimizer(model2, dataloader, valdataloader)

