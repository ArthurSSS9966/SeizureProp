import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

def constructDataset(data):
    '''
    This function constructs the dataset from the data file
    :param data: structure containing the fields "seizureNumber" and "data"
    :return: dataset structure containing the fields "seizureNumber", "units", "samplingRate", and "eegData"
    '''

    rawdata = data["data"]
    seizureNumber = data["seizureNumber"]

    # Process the txt file:

    # 1. Separate the section with a lot of % signs
    # 2. Extract the data from the section
    # 3. Save the data to a new file

    # Split the data by the % signs
    data = rawdata.split("%")

    # Remove empty sections
    data = [section for section in data if section != '']

    units = data[4].replace("\n", "").replace("\t", "").split(":")[-1]
    samplingRate = int(data[7].split()[2].split(".")[0])
    eegData = data[14]

    # Extract the data from eegData for each line
    eegData = eegData.split("\n")

    timeStamp = [line.split("\t")[0] for line in eegData][1:]
    # Remove all None values
    timeStamp = [ts for ts in timeStamp if ts != '' and ts != '--- BREAK IN DATA ---']

    # Convert the string timestamps to datetime objects
    datetime_objects = [datetime.strptime(ts, '%m/%d/%Y %H:%M:%S') for ts in timeStamp]

    # Define the start time (index 0 timestamp) and calculate the reference time
    start_time = datetime_objects[0]
    reference_time = start_time + timedelta(seconds=10)
    # Initialize an empty list for normalized times
    normalized_times = []

    # Calculate the normalized time in milliseconds relative to the reference time
    for i, dt in enumerate(datetime_objects):
        # Calculate the difference in seconds between the current time and the reference time
        time_diff_seconds = (dt - reference_time).total_seconds()

        # Determine the sample index within the second
        sample_index = i % samplingRate  # Assumes that samples are in order

        # Calculate the time in milliseconds for the sample within its second
        time_in_milliseconds = time_diff_seconds * 1000 + (sample_index * (1000 / samplingRate))

        normalized_times.append(time_in_milliseconds)

    normalized_times = np.array(normalized_times)

    # Separate the data by tabs, and excludes the first three tabs and the last tab
    eegData = [line.split("\t")[3:-1] for line in eegData][1:]

    # Convert the data to a DataFrame
    eegData = pd.DataFrame(eegData)

    # Delete the columns with "SHORT" value
    eegData = eegData.loc[:, (eegData != "SHORT").all()]

    # Replace the "AMPSAT" with maximum value
    eegData = eegData.replace("AMPSAT", 2000)

    # Delete the rows with None values
    eegData = eegData.dropna()

    # Convert the data to float
    eegData = np.array(eegData.astype(float))


    # Construct the structure of the dataset
    dataset = {
        "seizureNumber": seizureNumber,
        "units": units,
        "samplingRate": samplingRate,
        "normalized_times": normalized_times,
        "eegData": eegData
    }

    return dataset


DATA_FOLDER = "D:/Blcdata/seizure"
OUTPUT_FOLDER = "data"
PAT_NO = 65

if __name__ == "__main__":
    # Load the data
    data_folder = os.path.join(DATA_FOLDER, "P0{:02d}".format(PAT_NO))

    seizureData = []  # List to store the seizure data

    # Load the txt data file
    for file in os.listdir(data_folder):
        if file.endswith(".txt"):
            txt_file = os.path.join(data_folder, file)
            with open(txt_file, "r", encoding='utf-16') as f:
                data = f.read()
                # Extract the number from the file name
                seizureNumber = int(''.join(filter(str.isdigit, file)))

                seizuredatatep = {"seizureNumber": seizureNumber, "data": data}
                seizureData.append(seizuredatatep)

    for data in seizureData:
        dataset = constructDataset(data)
        output_dir = os.path.join(OUTPUT_FOLDER, f"P{PAT_NO}")

        # Save the data to a new file
        if not os.path.exists(output_dir):
            print(f"Creating directory {output_dir}")
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"seizure_{dataset['seizureNumber']}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(dataset, f)
            print(f"Data for seizure {dataset['seizureNumber']} saved to {output_file}")