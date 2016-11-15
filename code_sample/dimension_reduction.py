import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import Timer, MetaData

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
# Note that this is a numpy structured array as the data set contains both int and float
# http://docs.scipy.org/doc/numpy/user/basics.rec.html
activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

# convert to pandas data frame
df = pd.DataFrame(activityData)




# scale to min 0 max 1
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df.ix[:, 2:42])

# Perform PCA and explore first 3 components
print('Performing PCA ...')
pca = PCA(n_components=3)
dftr = pca.fit_transform(scaled_data)

print('Visualizing ... ')
plt.style.use("ggplot")

dftr = np.column_stack((dftr,df.subject))

# Code to sample 2000 points
# idx = np.random.randint(len(pltdata), size=2000)
# dftr = pltdata[idx,:]

# Plot 2d PCA
# fig = plt.figure(1, figsize=(12,12))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('principal component 1')
# ax.set_ylabel('principal component 2')
# plt.title('Top 2 Principal compoments')
# plt.scatter(dftr[:,0], dftr[:,1], c=dftr[:,3], alpha=0.5)

# Plot 3d PCA
fig = plt.figure(1, figsize=(12,12))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
ax.set_zlabel('principal component 3')
plt.title('Top 3 Principal components')
plt.scatter(dftr[:,0], dftr[:,1], dftr[:,2], c=dftr[:,3], marker='x', cmap=plt.cm.Accent)

plt.show()

