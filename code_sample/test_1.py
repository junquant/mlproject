# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:54:04 2016

@author: yizhang2013
"""

import os
import numpy as np

# Set the correct working directory to detect the files
currentDir = os.getcwd()
inputDir = '../data/protocol'

if currentDir != inputDir:
    os.chdir (inputDir)

# Create list with non-repeated column names - to append repeated ones later
"""
columnNames = ['timestamp(s)', 'activityID', 'heartRate(bpm)']

# Create list to be appended with body part
rawIMU = ['%s-temp(C)', '%s-3D-accel-16', '%s-3D-accel-16', '%s-3D-accel-16', '%s-3D-accel-6', '%s-3D-accel-6', '%s-3D-accel-6', '%s-3D-gyroscope', '%s-3D-gyroscope', '%s-3D-gyroscope', '%s-3D-magnetometer', '%s-3D-magnetometer', '%s-3D-magnetometer', '%s-orientation', '%s-orientation', '%s-orientation', '%s-orientation']

# Prefixes to be appended to rawIMU
prefix = ['hand','chest','ankle']

# Append prefixes
for l in range(0,3):
    for imu in rawIMU:
        imu = str(imu) % prefix[l]
        columnNames.append(imu)
"""
# Begin extracting data from .dat to .csv from each .dat file
activity_data = []
    
for i in range(101,110,1):
    fileName = 'subject%s' % i
    #read all protocal data into an array of arrays
    activity_data.append(np.loadtxt(fileName + '.dat'))        # Load .dat file
"""    
    outputCSV = fileName + '.csv'          # Create .csv file name
    
    df = pd.DataFrame(fileIn)       # Convert np.ndarray to pandas DataFrame

    df.columns = columnNames
    df.to_csv(outputCSV, encoding='utf-8', chunksize=1)
    
    fileIn.close()
 """
HB_IMU_Reading_raw=[]
nan_indice=[]
replace_indice=[]
for i in range(9):
    print('size of subject ',i, ': shape is ',activity_data[i].shape,', nan data has ',np.count_nonzero(np.isnan(activity_data[i][:,2:])))
    #remove the timestamp and activity ID, i.e. first two columns
    HB_IMU_Reading_raw.append(activity_data[i][:,2:])
    print('HB_IMU_Reading_raw ',i,' is ',HB_IMU_Reading_raw[i].shape)
    #removing nan records
    while np.count_nonzero(np.isnan(HB_IMU_Reading_raw[i])) > 0 :
        #find the indice matrix where the value is nan
        nan_indice = np.isnan(HB_IMU_Reading_raw[i])        
        print('    no. of nan is ',np.count_nonzero(nan_indice))
        test_array = nan_indice[0]
        test_array.shape=[1,52]
        #shift the indices matrics by one row up
        replace_indice=np.concatenate((nan_indice[1:],test_array))
        #replace the nan value with the value above or below
        HB_IMU_Reading_raw[i][nan_indice]= HB_IMU_Reading_raw[i][replace_indice]

#HB_IMU_Reading_raw is the array of 9 matrix, to be processed by PCA