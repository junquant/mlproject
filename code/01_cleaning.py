import numpy as np
import pandas as pd
from datetime import datetime


# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_10.txt'

# Utility class for timing the script
# --------------------------------------
class Timer:
    def __init__(self):
        pass

    def getTimeString(self, mydate):
        return str(mydate.strftime('%Y-%m-%d %H:%M:%S'))

    def getTime(self):
        return datetime.now()

print('------------------------------------------------------------')
print('Reading files ... ')
timer = Timer()
startTime = timer.getTime()
print('------------------------------------------------------------')

print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes


dataType = [('timestamp','float'),('activity_id','float'),('heartrate_bpm','float'),
            # imu hand - accel
            ('hand_temp_c','float'),
            ('hand_3d_accel_16g_1','float'),('hand_3d_accel_16g_2', 'float'),('hand_3d_accel_16g_3','float'),
            ('hand_3d_accel_6g_1','float'),('hand_3d_accel_6g_2', 'float'),('hand_3d_accel_6g_3','float'),
            # imu hand - gyro, magneto
            ('hand_3d_gyroscope_1','float'),('hand_3d_gyroscope_2','float'),('hand_3d_gyroscope_3','float'),
            ('hand_3d_magnetometer_1', 'float'),('hand_3d_magnetometer_2','float'),('hand_3d_magnetometer_3','float'),
            # imu hand - orientation
            ('hand_orientation_1','float'),('hand_orientation_2','float'), ('hand_orientation_3', 'float'),('hand_orientation_4','float'),
            # imu chest - accel
            ('chest_temp_c', 'float'),
            ('chest_3d_accel_16g_1', 'float'), ('chest_3d_accel_16g_2', 'float'), ('chest_3d_accel_16g_3', 'float'),
            ('chest_3d_accel_6g_1', 'float'), ('chest_3d_accel_6g_2', 'float'), ('chest_3d_accel_6g_3', 'float'),
            # imu chest - gyro, magneto
            ('chest_3d_gyroscope_1', 'float'), ('chest_3d_gyroscope_2', 'float'), ('chest_3d_gyroscope_3', 'float'),
            ('chest_3d_magnetometer_1', 'float'), ('chest_3d_magnetometer_2', 'float'),
            ('chest_3d_magnetometer_3', 'float'),
            # imu chest - orientation
            ('chest_orientation_1', 'float'), ('chest_orientation_2', 'float'), ('chest_orientation_3', 'float'),
            ('chest_orientation_4', 'float'),
            # imu ankle - accel
            ('ankle_temp_c', 'float'),
            ('ankle_3d_accel_16g_1', 'float'), ('ankle_3d_accel_16g_2', 'float'), ('ankle_3d_accel_16g_3', 'float'),
            ('ankle_3d_accel_6g_1', 'float'), ('ankle_3d_accel_6g_2', 'float'), ('ankle_3d_accel_6g_3', 'float'),
            # imu ankle - gyro, magneto
            ('ankle_3d_gyroscope_1', 'float'), ('ankle_3d_gyroscope_2', 'float'), ('ankle_3d_gyroscope_3', 'float'),
            ('ankle_3d_magnetometer_1', 'float'), ('ankle_3d_magnetometer_2', 'float'),
            ('ankle_3d_magnetometer_3', 'float'),
            ('subject', 'int')
            ]

# Note that this is numpy structured array as the data set contains both int and float
# http://docs.scipy.org/doc/numpy/user/basics.rec.html
activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

# convert to pandas data frame
df = pd.DataFrame(activityData)

# count missing values in df
print('Number of missing values in data frame')
print('--------------------------------------')
print(df.isnull().sum())

# populate next missing value with last valid observation
df = df.fillna(method='ffill')

# populate previous missing values with mean of the subj

    # work in progress

# Perform a summary of the data
print('Summary of data frame')
print('--------------------------------------')
print(df.describe())



# Visualization

# Impute using mean

# PCA

# Print end time
print('End Time : ',timer.getTime())
