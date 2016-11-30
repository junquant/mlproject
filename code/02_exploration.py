import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import Timer, MetaData


# File properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_all.txt'
plotDir = '../plots/'

# Read Data
# -----------------------------------------------------

metadata = MetaData()
dataType = metadata.getProcessedColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
data = np.loadtxt(filePath, delimiter = ',', skiprows=1, dtype=dataType)
df = pd.DataFrame(data)

subj = df.ix[:,-2]
activity = df.ix[:,-1]
subj_activity = (100*subj) + activity
df = pd.concat([df,subj_activity],axis=1)
df.rename(columns={0:'activity_subj'}, inplace=True)

readings = df.iloc[:,:-3]


# Sample Data for Exploration
# ---------------------------------------------

strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.99, test_size=0.01, random_state=2016)

# stratify based on subj_activity
for big_index, small_index in strat_split.split(df,subj_activity):
    df_big, df_small = df.ix[big_index], df.ix[small_index]
    print('Size of data set: ', len(df))
    print('Size of training data set: ', len(big_index))
    print('Size of test data set: ', len(small_index))

df_small_readings = df_small.iloc[:,:-3]
df_small_subject = df_small.iloc[:,-2]
df_small_activity = df_small.iloc[:,-1]


# PCA Exploration
# ---------------------------------------------
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df_small_readings)

print('Plotting PCA 2 components ...')

pca = PCA()

dftr = pca.fit_transform(scaled_data)

# Save PCA components to csv
header = [x[0] for x in dataType][:-2]
header.insert(0, 'component')
header_str = ','.join(header)
pca_data = np.insert(pca.components_[:3], 0, [1,2,3], axis=1)
pca_table = pd.DataFrame(data=pca_data, columns=header)
fig, ax = plt.subplots(1,1)
ax.get_xaxis().set_visible(False)
df.plot(table=pca_table, ax=ax)  # TODO: plot table
plt.show()
np.savetxt('../exploration/pca_components.csv', pca_data, delimiter=',', header=header_str)