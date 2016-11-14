import numpy as np
import pandas as pd
from code_utilities.custom_utilities import Timer, MetaData

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_10.txt'
outputFile = '../data/consolidated_clean.txt'

metadata = MetaData()
dataType = metadata.getOriginalColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
# Note that this is a numpy structured array as the data set contains both int and float
# http://docs.scipy.org/doc/numpy/user/basics.rec.html
activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

# convert to pandas data frame
df = pd.DataFrame(activityData)

# count missing values in df
print('--------------------------------------')
print('Number of missing values in data frame')
print('--------------------------------------')
print(df.describe())
print(df.isnull().sum())

# remove unnecessary columns
nonOrientationCols = [col for col in df.columns if 'orientation' not in col]
df = df[nonOrientationCols]

# populate next missing value with last valid observation
df = df.fillna(method='ffill')

# populate previous missing values next valid observation
df = df.fillna(method='bfill')

# Perform a summary of the data
print('--------------------------------------')
print('Summary of data frame after replacement')
print('--------------------------------------')
print(df.describe())
print(df.isnull().sum())

df.to_csv(outputFile, header=True, index=None,sep=',')