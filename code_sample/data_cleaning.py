# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:54:04 2016

@author: thomasjiang
"""

import os
import numpy as np
import pandas as pd

# Set the correct working directory to detect the files
currentDir = os.getcwd()
inputDir = '../data/protocol'

if currentDir != inputDir:
    os.chdir (inputDir)

# Begin extracting data from .dat to .csv from each .dat file
    
for i in range(101,110,1):
    fileName = 'subject%s' % i
    fileIn = np.loadtxt(fileName + '.dat')        # Load .dat file
    outputCSV = fileName + '.csv'          # Create .csv file name
    
    df = pd.DataFrame(fileIn)       # Convert np.ndarray to pandas DataFrame
    
    # Create list with non-repeated column names - to append repeated ones later
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
    
    df.columns = columnNames
    df.to_csv(outputCSV, encoding='utf-8', chunksize=1)
    
    fileOut.close()
             
    
    
    
