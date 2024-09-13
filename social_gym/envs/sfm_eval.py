# Run this file when you have data from both runs - self.useRobots = True and self.useRobots = True

# Necessary imports
import pandas as pd
import numpy as np
import os
import similaritymeasures 

# Directory of all the log files
trajDir = 'home/put_here'

# Get list of file and sort it. Then we need to always access two at a time for comparison
files = os.listdir(trajDir)
files.sort()
filerange = int(len(files)/2)

# Function to convert string to numpy array
def str_to_array(pos_str):
    # Remove the brackets and extra spaces, then split by space
    pos_list = pos_str.replace('[', '').replace(']', '').split()
    # Convert list of strings to numpy array of floats
    return np.array(pos_list, dtype=float)

def get_file_pair(id, directory, filelist):
    file0 = ''
    file1 = ''
    for file in filelist:
        porridge = file.split('_')
        if porridge[2] == f'ep{id}':
            if porridge[3] == '0.txt':
                file0 = directory+'/'+file
            else:
                file1 = directory+'/'+file
    columnNames = ['PedestrianID','Position','RobotDistance']
    df0 = pd.read_csv(file0, names=columnNames)
    df1 = pd.read_csv(file1, names=columnNames)

    # Apply the conversion function to the Position column and RobotDistance column
    df0['Position'] = df0['Position'].apply(str_to_array)
    df0['RobotDistance'] = pd.to_numeric(df0['RobotDistance'])

    df1['Position'] = df1['Position'].apply(str_to_array)
    df1['RobotDistance'] = pd.to_numeric(df1['RobotDistance'])

    return df0, df1

def calculate_trajectory_length(df):
    if len(df) < 2:
        return 0  # No trajectory to measure
    # Calculate differences between consecutive points
    diffs = np.diff(df['Position'].to_list(), axis=0)
    # Calculate Euclidean distances
    distances = np.linalg.norm(diffs, axis=1)
    # Sum the distances
    total_length = np.sum(distances)
    return total_length


def calculate_total_time(df):
    # Time interval between each point in seconds
    time_interval = 0.067
    num_points = len(df)
    if num_points < 2:
        return 0  # No intervals if less than two points
    num_intervals = num_points - 1
    total_time = num_intervals * time_interval
    return total_time

trajectoryLengths0 = []
trajectoryLengths1 = []
trajectoryTimes0 = []
trajectoryTimes1 = []
minDistances0 = []
minDistances1 = []
frechetDistances = []

frachetForSignificance = []
lengthForSignificance = []
timeForSignificance = []
minDistanceForSignificance = []

for i in range(filerange):
    # Get the dataframes for both files
    df0, df1 = get_file_pair(i, trajDir, files)

    # Get pedestrian list
    pedestrians = df0.PedestrianID.unique().tolist()

    # Set minimum distance to maximum value to make it easier to compare and store
    minDist0 = np.inf
    minDist1 = np.inf

    ped_lengths0 = []
    ped_times0 = []
    ped_lengths1 = []
    ped_times1 = []
    fr_distances = []
    # Go through each pedestrian in the acquired files
    for pedestrian in pedestrians:
        trajectory0 = df0[df0.PedestrianID==pedestrian]
        trajectory1 = df1[df1.PedestrianID==pedestrian]

        length0 = calculate_trajectory_length(trajectory0)
        time0 = calculate_total_time(trajectory0)
        mindistance0 = trajectory0['RobotDistance'].min()
        if mindistance0 < minDist0:
            minDist0 = mindistance0
        ped_lengths0.append(length0)
        ped_times0.append(time0)

        length1 = calculate_trajectory_length(trajectory1)
        time1 = calculate_total_time(trajectory1)
        mindistance1 = trajectory1['RobotDistance'].min()
        if mindistance1 < minDist1:
            minDist1 = mindistance1
        ped_lengths1.append(length1)
        ped_times1.append(time1)
        lengthForSignificance.append(length1)
        timeForSignificance.append(time1)
        minDistanceForSignificance.append(minDist1)

        traj0 = np.array(trajectory0['Position'].values.tolist())
        traj1 = np.array(trajectory1['Position'].values.tolist())

        fr_dist = similaritymeasures.frechet_dist(traj0,traj1)
        fr_distances.append(fr_dist)
        frachetForSignificance.append(fr_dist)

    # Append all lengths
    trajectoryLengths0.append(np.mean(ped_lengths0))
    trajectoryLengths1.append(np.mean(ped_lengths1))

    # Append all times
    trajectoryTimes0.append(np.mean(ped_times0))
    trajectoryTimes1.append(np.mean(ped_times1))

    # Append Minimum Distances of both cases to the list
    minDistances0.append(minDist0)
    minDistances1.append(minDist1)

    # Append Frechet Distances
    frechetDistances.append(np.mean(fr_distances))

print('Trajectory Lengths 0: ',np.mean(trajectoryLengths0), np.std(trajectoryLengths0))
print('Trajectory Times 0: ', np.mean(trajectoryTimes0), np.std(trajectoryTimes0))
print('Minimum Distances 0: ', np.mean(minDistances0), np.std(minDistances0))

print('Trajectory Lengths 1: ',np.mean(trajectoryLengths1), np.std(trajectoryLengths1))
print('Trajectory Times 1: ', np.mean(trajectoryTimes1), np.std(trajectoryTimes1))
print('Minimum Distances 1: ', np.mean(minDistances1), np.std(minDistances1))

print('Frechet Distance: ', np.mean(frechetDistances), np.std(frechetDistances))

with open('Data_For_Significance.csv', 'w') as f:
    for a,b,c,d in zip(frachetForSignificance,lengthForSignificance,minDistanceForSignificance,timeForSignificance):
        f.write(f"{a},{b},{c},{d}\n")

print("DONE!")
