# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:54:04 2016

@author: thomasjiang
"""

import os
import numpy as np
import pandas as pd

currentDir = os.getcwd()

if currentDir != 'C:\\Users\\thomasjiang\\Desktop\\Google Drive\\SMU MITB\\AY2016-2017\\S1\\ISSS610 Applied Machine Learning\\ml-project\\PAMAP2_Dataset\\Protocol':
    os.chdir ('C:\\Users\\thomasjiang\\Desktop\\Google Drive\\SMU MITB\\AY2016-2017\\S1\\ISSS610 Applied Machine Learning\\ml-project\\PAMAP2_Dataset\\Protocol')
    
for i in range(101,110,1):
    fileName = 'subject%s' % i
    fileIn = np.loadtxt(fileName + '.dat')        # Load .dat file
    df = pd.DataFrame(fileIn)       # Convert np.ndarray to pandas DataFrame
    
    columnNames = ['timestamp(s)', 'activityID', 'heartRate(bpm)']
    
    rawIMU = ['%s-temp(C)', '%s-3D-accel-16', '%s-3D-accel-16', '%s-3D-accel-16', '%s-3D-accel-6', '%s-3D-accel-6', '%s-3D-accel-6', '%s-3D-gyroscope', '%s-3D-gyroscope', '%s-3D-gyroscope', '%s-3D-magnetometer', '%s-3D-magnetometer', '%s-3D-magnetometer', '%s-orientation', '%s-orientation', '%s-orientation', '%s-orientation']
    
    prefix = ['hand','chest','ankle']
    
    
    for l in range(0,3):
        for imu in rawIMU:
            imu = str(imu) % prefix[l]
            columnNames.append(imu)
    
    df.columns = columnNames
    
    csvFileName = fileName + '.csv'    
    
    df.to_csv(csvFileName, sep='\t', encoding = 'utf-8')