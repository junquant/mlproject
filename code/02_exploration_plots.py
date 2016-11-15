import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import Timer, MetaData

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_101.txt'

# Plot functions
# -----------------------------------------------------
def plot_correlation(dataframe, title=''):
    lang_names = dataframe.columns.tolist()
    tick_indices = np.arange(0.5, len(lang_names) + 0.5)
    plt.figure(figsize=(12, 9))
    plt.pcolor(dataframe.values, cmap='RdBu', vmin=-1, vmax=1)
    colorbar = plt.colorbar()
    colorbar.set_label('Correlation')
    plt.title(title)
    plt.xticks(tick_indices, lang_names, rotation='vertical')
    plt.yticks(tick_indices, lang_names)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

def plot_scree(explainedVarianceRatio, colsNumber, title=''):
    pc = np.arange(colsNumber) + 1

    plt.figure(figsize=(12, 9))
    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('% of Variance Explained')
    plt.plot(pc, explainedVarianceRatio)


# Code Start
# -----------------------------------------------------

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
activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

# convert to pandas data frame
df = pd.DataFrame(activityData)

print(df.describe())

# Correlation matrix
# ---------------------------------------------
plt.style.use('ggplot')
dfReadings = df.iloc[:, 2:-1]
corrPlot = plot_correlation(dfReadings.corr(), title='IMU readings')


# PCA
# ---------------------------------------------
# scale to min 0 max 1
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df.ix[:, 2:42])


# Perform PCA and explore first 3 components
print('Performing PCA ...')
pca = PCA()
dftr = pca.fit_transform(scaled_data)

print('Visualizing ... ')
plt.style.use("ggplot")
dftr = np.column_stack((dftr[:,0:3],df.activity_id))

# Code to sample 2000 points (uncomment to sample 2000 pts
# ---------------------------------------------
# idx = np.random.randint(len(pltdata), size=2000)
# dftr = pltdata[idx,:]

# Plot 2d PCA

# fig = plt.figure(figsize=(12,12))
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

# Get principal components and explained variance ration
comp = pca.components_
evr = pca.explained_variance_ratio_

screePlot = plot_scree(evr, len(evr), 'Scree Plot')

# Show the graphs
plt.show()

# Print end time
print('End Time : ', timer.getTime())

