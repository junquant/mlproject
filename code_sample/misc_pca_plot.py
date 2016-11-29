import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from utilities import Timer, MetaData, ResultsWriter

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_all.txt'

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

# Separating the subject and activity
subj = df.ix[:,-2]
activity = df.ix[:,-1]
subj_activity = ((100*subj) + activity).astype(int)
df = pd.concat([df,subj_activity],axis=1)
df.rename(columns={0:'activity_subj'}, inplace=True)

# step 1.2 - scale to min 0 max 1 and Perform PCA
print('Performing PCA ...')
minmax_scaler = MinMaxScaler()
pca = PCA(n_components=10)

minmax_df = minmax_scaler.fit_transform(df.ix[:,:-3])
pca_df = pca.fit_transform(minmax_df)

print('PCA Components: ', pca.components_)

select = pca_df[:,:3]
select = pd.DataFrame(select)
df = pd.concat([select, subj_activity], axis=1)

# Get a subset of the data
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.999, test_size=0.001, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,df.iloc[:,-1]):
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

fignum = 1
fig = plt.figure(fignum,figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c=x[:,3] ,marker='o')
plt.show()