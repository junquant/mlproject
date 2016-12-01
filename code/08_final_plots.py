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
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.95, test_size=0.05, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,df['predicted_subj_activity']):
    df_large, df_small = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of large data set: ', len(train_index))
    print('Size of small data set: ', len(test_index))

df_small_readings = df_small.ix[:,:-6]
df_small_true_subj = df_small.ix[:,-6]
df_small_true_activity = df_small.ix[:,-5]
df_small_true_subj_activity = df_small.ix[:,-4]

df_small_pred_subj = df_small.ix[:,-2]
df_small_pred_activity = df_small.ix[:,-1]
df_small_pred_subj_activity = df_small.ix[:,-3]

accurate = np.array(df_small_true_subj_activity == df_small_pred_subj_activity)

plt.style.use('ggplot')
# --------------------------------------------------------
# Correct vs Incorrect Plots

minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df_small_readings)

print('Plotting PCA 2 components ...')

pca = PCA()

dftr = pca.fit_transform(scaled_data)
dftr = np.column_stack((dftr[:,0:3],df_small_true_subj_activity))

fig = plt.figure(2, figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.title('Top 2 Principal Components')
plt.scatter(dftr[accurate][:,0], dftr[accurate][:,1],
            c=dftr[accurate][:,3], alpha=0.5, cmap=plt.cm.prism, marker='o')
plt.scatter(dftr[~accurate][:,0], dftr[~accurate][:,1],
            c=dftr[~accurate][:,3], alpha=1, cmap=plt.cm.prism, marker='v', s= 50)

plt.savefig('../plots/pca_2components_classified.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Plotting PCA 3 components ...')

fig = plt.figure(3, figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.view_init(-142,162)
plt.title('Top 3 Principal Components')
ax.scatter(dftr[accurate][:,0], dftr[accurate][:,1], dftr[accurate][:,2],
           c=dftr[accurate][:,3],cmap=plt.cm.prism, marker='o', alpha=.4)
ax.scatter(dftr[~accurate][:,0], dftr[~accurate][:,1], dftr[~accurate][:,2],
           c=dftr[~accurate][:,3],cmap=plt.cm.prism, marker='v', alpha=1, s= 50)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

plt.savefig('../plots/pca_3components_classified.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Hand, Chest, Ankle plot ...')

df_hca = df_small_readings.as_matrix()
df_tsa = df_small_true_subj_activity.as_matrix()

fig = plt.figure(4, figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('hand_temp_c')
ax.set_ylabel('chest_temp_c')
ax.set_zlabel('ankle_temp_c')
ax.view_init(92,50)
plt.title('Hand Chest Ankle Temperature')
ax.scatter(df_hca[accurate][:,1], df_hca[accurate][:,11], df_hca[accurate][:,21],
           c=df_tsa[accurate],cmap=plt.cm.prism, marker='o', alpha=.1)
ax.scatter(df_hca[~accurate][:,1], df_hca[~accurate][:,11], df_hca[~accurate][:,21],
           c=df_tsa[~accurate],cmap=plt.cm.prism, marker='v', alpha=1, s= 50)

plt.savefig('../plots/hca_temp_classified.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

plt.show()