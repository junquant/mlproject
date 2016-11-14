import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import functools
from code_utilities.custom_utilities import Timer, MetaData

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean.txt'

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

print(activityData)

# convert to pandas data frame
df = pd.DataFrame(activityData)
print(df.describe())
# Perform PCA and explore first 3 components
pca = PCA()
dftr = pca.fit_transform(df.ix[:, 2:42])
pca_comp = pca.components_
print('PCA Components:')
print(pca_comp)
print('--------------------------------------')

