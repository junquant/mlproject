import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

# Perform PCA and explore first 3 components
print('Performing PCA ...')
pca = PCA(n_components=3)
dftr = pca.fit_transform(df.ix[:, 2:42])

print('Visualizing ... ')
plt.style.use("ggplot")

fig = plt.figure(1, figsize=(12,12))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
plt.title('Top 2 Principal compoments')
plt.scatter(dftr[:,0], dftr[:,1], c=df.activity_id)

# fig = plt.figure(1, figsize=(12,12))
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.set_xlabel('principal component 1')
# ax.set_ylabel('principal component 2')
# ax.set_zlabel('principal component 3')
# plt.title('Top 3 Principal components')
# plt.scatter(dftr[:,0], dftr[:,1], dftr[:,2], c=df.subject, marker='x', )

plt.show()

