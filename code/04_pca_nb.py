import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from utilities import Timer, MetaData

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_101.txt'

metadata = MetaData()
dataType = metadata.getProcessedColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
# Note that this is a numpy structured array as the data set contains both int and float
# http://docs.scipy.org/doc/numpy/user/basics.rec.html
data = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

df = pd.DataFrame(data)
readings = df.ix[:,2:42]
subj = df.ix[:,-1]
activity = df.ix[:,1]

# PCA
# ---------------------
# scale to min 0 max 1
print('Performing PCA ...')
minmax_scaler = MinMaxScaler()
pca = PCA()

data_scaled = minmax_scaler.fit_transform(readings)
data_pca = pca.fit_transform(data_scaled)

# split data set into test and train
# ---------------------
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=2016)

for train_index, test_index in strat_split.split(readings,activity):
    readings_train, readings_test = readings.ix[train_index], readings.ix[test_index]
    activity_train, activity_test = activity.ix[train_index], activity.ix[test_index]
    print('Size of data set: ', len(readings))
    print('Size of training data set: ', len(train_index))
    print('Size of test data set: ', len(test_index))

train_table = pd.concat([readings_train,activity_train], axis= 1)
test_table = pd.concat([readings_test,activity_test], axis= 1)

print(train_table.ix[:,-2:].groupby('activity_id').count())
print(test_table.ix[:,-2:].groupby('activity_id').count())

# print(x)
# x = pd.concat([readings_train,subj_train], axis=1)
#
# print(pd.crosstab(x.ix[:,1], x.ix[:,-1]))

# ---------------------
# Grid Search Predict A

# Join Predict to result

# Grid Search Predict B

# Calc score for A + B
# ---------------------

# ---------------------
# Grid Search Predict B

# Join Predict to result

# Grid Search Predict A

# Calc score for A + B
# ---------------------

# output to results folder

