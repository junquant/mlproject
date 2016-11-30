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


# PCA Exploration
# ---------------------------------------------
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(readings)

print('Plotting PCA 2 components ...')

pca = PCA()

dftr = pca.fit_transform(scaled_data)

# Save PCA components to csv
features = [x[0] for x in dataType][:-2]
pca_table = pd.DataFrame(data=pca.components_[:3]).transpose()
from pandas.tools.plotting import table
plt.figure(figsize=(10,8))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table(ax, pca_table, loc='center', colLabels=('PC1','PC2','PC3'), rowLabels=features)
plt.savefig('../exploration/pca_table.png')

pc1 = sorted(pca.components_[0], reverse=True)
