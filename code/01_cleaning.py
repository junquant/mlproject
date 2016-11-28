import numpy as np
import pandas as pd

from utilities import Timer, MetaData

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_all.txt'
outputFile = '../data/consolidated_clean_all.txt'

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
#activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)
activityData = np.loadtxt(filePath, delimiter = ',', skiprows=1, dtype=dataType)
print('loading Time : ', timer.getTime())
# convert to pandas data frame
df = pd.DataFrame(activityData)

# count missing values in df
print('--------------------------------------')
print('Number of missing values in data frame')
print('--------------------------------------')
print(df.shape)
print(df.isnull().sum())

# remove unnecessary columns
nonOrientationCols = [col for col in df.columns if 'orientation' not in col]
nonSaturationCols = [col for col in nonOrientationCols if '_accel_6g_' not in col]
df = df[nonSaturationCols]

# for each subject, perform the following
for subj in range(101,109):
    print('Cleaning subj ', subj)
    subj_df = df.loc[df['subject'] == subj]

    # populate next missing value with last valid observation then
    # populate previous missing values next valid observation
    subj_df = subj_df.fillna(method='ffill')
    subj_df = subj_df.fillna(method='bfill')

    if subj == 101:
        clean_df = subj_df
    else:
        if subj == 104: # fix that 1 record with error
            subj_df = subj_df[subj_df.activity_id != 5]

        clean_df = clean_df.append(subj_df)

    # uncomment to write individual subj files
    # subj_df.to_csv(str(subj) + '.txt', header=True, index=None, sep=',')
    print(subj_df.shape)

# rearrange the cols - subj, activity, readings ....
cols = clean_df.columns.tolist()
cols = cols[2:-1] + [cols[-1]] + [cols[1]]
clean_df = clean_df[cols]

# Perform a summary of the data
print('--------------------------------------')
print('Summary of data frame after replacement')
print('--------------------------------------')
print(clean_df.shape)
print(clean_df.describe())
print(clean_df.isnull().sum())

clean_df.to_csv(outputFile, header=True, index=None, sep=',')

print('Finish Time : ', timer.getTime())
