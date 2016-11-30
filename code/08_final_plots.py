import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.98, test_size=0.02, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,df['predicted_subj_activity']):
    df_large, df_small = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of large data set: ', len(train_index))
    print('Size of small data set: ', len(test_index))

df_small_readings = df_small.iloc[:,:-6]
df_small_act_subj = df_small.iloc[:,-6]
df_small_act_activity = df_small.iloc[:,-5]
df_small_act_subj_activity = df_small.iloc[:,-4]

df_small_pred_subj = df_small.iloc[:,-3]
df_small_pred_activity = df_small.iloc[:,-2]
df_small_pred_subj_activity = df_small.iloc[:,-1]

# --------------------------------------------------------
# Correct vs Incorrect Plots

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
