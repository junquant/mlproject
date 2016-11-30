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
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.98, test_size=0.02, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,df['predicted_subj_activity']):
    df_large, df_small = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of large data set: ', len(train_index))
    print('Size of small data set: ', len(test_index))

# --------------------------------------------------------
# Work your plotting magic here
# You can choose to use df_large or df_small

x = df_small.as_matrix()
print(x)
print(x.shape)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")

# plt.figure(figsize=(10, 10))
# fig = plt.subplot(1, 1, 1, projection='3d')
# fig.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:,- 5], cmap=plt.cm.Spectral)
# fig.view_init(10, -72)
# plt.show()

fignum = 1
fig = plt.figure(fignum,figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c=x[:,-5] ,marker='o')
plt.show()

# ax = fig.add_subplot(4, 4, axpos)
# annotateStr = 'eps: %.1f | mins: %d \n# outliers: %d \n# clusters: %d \navg size of clusters: %.2f \nhomogeneity: %.2f \ncompleteness: %.2f' \
#               % (e, mins, num_outliers, num_clusters, avg_size, homogeneity, completeness)
# ax.annotate(annotateStr, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9)
# ax.scatter(x[:, 0], x[:, 1], c=label, cmap=plt.cm.Accent)
# axpos += 1
