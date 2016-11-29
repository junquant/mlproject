import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from utilities import Timer, MetaData, ResultsWriter

# file properties
# -----------------------------------------------------
filePath = '../data/results.txt'

metadata = MetaData()
dataType = metadata.getResultColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
data = np.loadtxt(filePath, delimiter = ',', skiprows = 1, dtype=dataType)
df = pd.DataFrame(data)

# Separating the subject and activity
subject = df.ix[:,-1]%100
subject.name = 'predicted_subj'
activity = (df.ix[:,-1] - subject) / 100
activity.name = 'predicted_activity'

df = pd.concat([df,subject,activity], axis=1)

# Get a subset of the data
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,df['predicted_subj_activity']):
    df_large, df_small = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of large data set: ', len(train_index))
    print('Size of small data set: ', len(test_index))

# --------------------------------------------------------
# Work your plotting magic here
# You can choose to use df_large or df_small
