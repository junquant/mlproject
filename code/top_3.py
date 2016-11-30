import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.kde import KernelDensity

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
df_small_subject = df_small.iloc[:,-3]
df_small_activity = df_small.iloc[:,-2]

# Top 3 Variables Plot
chest_temp = df_small.ix[:, 11]
hand_temp = df_small.ix[:, 1]
ankle_temp = df_small.ix[:, 21]
labels = df_small.ix[:,-1]
top_three = pd.concat([chest_temp, hand_temp, ankle_temp, labels],axis=1)

fig = plt.figure(8, figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('chest_temp')
ax.set_ylabel('hand_temp')
ax.set_zlabel('ankle_temp')
plt.title('Top 3 Variables')
ax.scatter(top_three.ix[:,0], top_three.ix[:,1], top_three.ix[:,2], c=top_three.ix[:,3], marker='o', cmap=plt.cm.prism)
plt.savefig('../plots/top_3variables.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)
plt.show()