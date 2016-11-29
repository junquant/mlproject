import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import Timer, MetaData, Plotter

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_all.txt'
plotDir = '../plots/'

# Plots
# -----------------------------------------------------
plotter = Plotter()

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

# Correlation matrix
# ---------------------------------------------
plt.style.use('ggplot')

dfReadings = df.iloc[:, :-3]
corrPlot = plotter.plot_correlation(dfReadings.corr(), title='IMU readings')


# PCA
# ---------------------------------------------
# scale to min 0 max 1
#minmax_scaler = MinMaxScaler()
#scaled_data = minmax_scaler.fit_transform(df.ix[:, :-2])


# Perform PCA and explore first 3 components
print('Performing PCA ...')
pca = PCA()
#dftr = pca.fit_transform(scaled_data)
dftr = pca.fit_transform(df.ix[:, :-3])

print('Visualizing ... ')
plt.style.use("ggplot")
dftr = np.column_stack((dftr[:,0:3],df.activity_id))

# Code to sample 2000 points (uncomment to sample 2000 pts
# ---------------------------------------------
idx = np.random.randint(len(df.activity_id), size=20000)
# dftr = pltdata[idx,:]

# Plot 2d PCA
# fig = plt.figure(figsize=(12,12))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('principal component 1')
# ax.set_ylabel('principal component 2')
# plt.title('Top 2 Principal compoments')
# plt.scatter(dftr[:,0], dftr[:,1], c=dftr[:,3], alpha=0.5)

# Plot 3d PCA
# Sample 500 points from each class and plot it
# ----------------------------------------------
# dftr = xxxx

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
ax.set_zlabel('principal component 3')
plt.title('Top 3 Principal components')
#for i in MetaData.activities:
#    np.random.choice(np.indices)
#plt.scatter(dftr[:,0], dftr[:,1], dftr[:,2], c=dftr[:,3], marker='x', cmap=plt.cm.Accent)
plt.scatter(dftr[idx,0], dftr[idx,1], dftr[idx,2], c=dftr[idx,3],marker='x', cmap=plt.cm.prism)
fig2 = plt.figure(figsize=(12,12))
ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
plt.title('Top 2 Principal components')
plt.scatter(dftr[idx,0], dftr[idx,1], c=dftr[idx,3],marker='x', cmap=plt.cm.prism)

#plt.scatter(dftr[idx,0], dftr[idx,1], dftr[idx,2], c=dftr[idx,3], cmap=plt.cm.rainbow)
# Get principal components and explained variance ration
comp = pca.components_
evr = pca.explained_variance_ratio_

screePlot = plot_scree(evr, len(evr), 'Scree Plot')

# Show the graphs
plt.show()

# Print end time
print('End Time : ', timer.getTime())

