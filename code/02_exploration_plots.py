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

# Correlation Matrix
# ---------------------------------------------
plt.style.use('ggplot')

print('Plotting correlation ...')
plotdata = readings.corr()
lang_names = plotdata.columns.tolist()
tick_indices = np.arange(0.5, len(lang_names) + 0.5)

plt.figure(1,figsize=(10, 10))
plt.pcolor(plotdata.values, cmap='RdBu', vmin=-1, vmax=1)
colorbar = plt.colorbar()
colorbar.set_label('Correlation')
plt.title('IMU Readings')
plt.xticks(tick_indices, lang_names, rotation='vertical')
plt.yticks(tick_indices, lang_names)
plt.xlim(0,len(lang_names))
plt.ylim(0,len(lang_names))
plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

plt.savefig('../plots/correlation.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# Sample Data for Exploration
# ---------------------------------------------

strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.99, test_size=0.01, random_state=2016)

# stratify based on subj_activity
for big_index, small_index in strat_split.split(df,subj_activity):
    df_big, df_small = df.ix[big_index], df.ix[small_index]
    print('Size of data set: ', len(df))
    print('Size of training data set: ', len(big_index))
    print('Size of test data set: ', len(small_index))

# PCA Exploration
# ---------------------------------------------
df_small_readings = df_small.iloc[:,:-3]
df_small_subject = df_small.iloc[:,-2]
df_small_activity = df_small.iloc[:,-1]

minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df_small_readings)

print('Plotting PCA 2 components ...')

pca = PCA()
dftr = pca.fit_transform(scaled_data)

dftr = np.column_stack((dftr[:,0:3],df_small_activity))

fig = plt.figure(2, figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.title('Top 2 Principal Components')
plt.scatter(dftr[:,0], dftr[:,1], c=dftr[:,3], alpha=0.5, cmap=plt.cm.prism)


plt.savefig('../plots/pca_2components.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Plotting PCA 3 components ...')

fig = plt.figure(3, figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('Top 3 Principal Components')
ax.scatter(dftr[:,0], dftr[:,1], dftr[:,2], c=dftr[:,3],marker='o', cmap=plt.cm.prism)

plt.savefig('../plots/pca_3components.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Plotting Scree Plot ...')

comp = pca.components_
evr = pca.explained_variance_ratio_

pc = np.arange(len(comp)) + 1
plt.figure(figsize=(12, 9))
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('% of Variance Explained')
plt.plot(pc, evr)

plt.savefig('../plots/screeplot.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# Variable Exploration
# ---------------------------------------------

print('End Time : ', timer.getTime())

plt.show()
