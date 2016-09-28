import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
import functools

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated.txt'

# Utility class for timing the script
# --------------------------------------
class Timer:
    def __init__(self):
        pass
    def getTimeString(self, mydate):
        return str(mydate.strftime('%Y-%m-%d %H:%M:%S'))
    def getTime(self):
        return datetime.now()

print('------------------------------------------------------------')
print('Reading files ... ')
timer = Timer()
startTime = timer.getTime()
print('------------------------------------------------------------')

print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

dataType = [('timestamp','float'),('activity_id','float'),('heartrate_bpm','float'),
            # imu hand - accel
            ('hand_temp_c','float'),
            ('hand_3d_accel_16g_1','float'),('hand_3d_accel_16g_2', 'float'),('hand_3d_accel_16g_3','float'),
            ('hand_3d_accel_6g_1','float'),('hand_3d_accel_6g_2', 'float'),('hand_3d_accel_6g_3','float'),
            # imu hand - gyro, magneto
            ('hand_3d_gyroscope_1','float'),('hand_3d_gyroscope_2','float'),('hand_3d_gyroscope_3','float'),
            ('hand_3d_magnetometer_1', 'float'),('hand_3d_magnetometer_2','float'),('hand_3d_magnetometer_3','float'),
            # imu hand - orientation
            ('hand_orientation_1','float'),('hand_orientation_2','float'), ('hand_orientation_3', 'float'),('hand_orientation_4','float'),
            # imu chest - accel
            ('chest_temp_c', 'float'),
            ('chest_3d_accel_16g_1', 'float'), ('chest_3d_accel_16g_2', 'float'), ('chest_3d_accel_16g_3', 'float'),
            ('chest_3d_accel_6g_1', 'float'), ('chest_3d_accel_6g_2', 'float'), ('chest_3d_accel_6g_3', 'float'),
            # imu chest - gyro, magneto
            ('chest_3d_gyroscope_1', 'float'), ('chest_3d_gyroscope_2', 'float'), ('chest_3d_gyroscope_3', 'float'),
            ('chest_3d_magnetometer_1', 'float'), ('chest_3d_magnetometer_2', 'float'),
            ('chest_3d_magnetometer_3', 'float'),
            # imu chest - orientation
            ('chest_orientation_1', 'float'), ('chest_orientation_2', 'float'), ('chest_orientation_3', 'float'),
            ('chest_orientation_4', 'float'),
            # imu ankle - accel
            ('ankle_temp_c', 'float'),
            ('ankle_3d_accel_16g_1', 'float'), ('ankle_3d_accel_16g_2', 'float'), ('ankle_3d_accel_16g_3', 'float'),
            ('ankle_3d_accel_6g_1', 'float'), ('ankle_3d_accel_6g_2', 'float'), ('ankle_3d_accel_6g_3', 'float'),
            # imu ankle - gyro, magneto
            ('ankle_3d_gyroscope_1', 'float'), ('ankle_3d_gyroscope_2', 'float'), ('ankle_3d_gyroscope_3', 'float'),
            ('ankle_3d_magnetometer_1', 'float'), ('ankle_3d_magnetometer_2', 'float'),
            ('ankle_3d_magnetometer_3', 'float'),
            # imu ankle - orientation
            ('ankle_orientation_1', 'float'), ('ankle_orientation_2', 'float'), ('ankle_orientation_3', 'float'),
            ('ankle_orientation_4', 'float'),
            ('subject', 'int')
            ]

# Note that this is a numpy structured array as the data set contains both int and float
# http://docs.scipy.org/doc/numpy/user/basics.rec.html
activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

# convert to pandas data frame
df = pd.DataFrame(activityData)

print(df.shape)
# count missing values in df
print('--------------------------------------')
print('Number of missing values in data frame')
print('--------------------------------------')
print(df.isnull().sum())

# remove unnecessary columns
nonOrientationCols = [col for col in df.columns if 'orientation' not in col]
df = df[nonOrientationCols]

# populate next missing value with last valid observation
df = df.fillna(method='ffill')

# populate previous missing values next valid observation
df = df.fillna(method='bfill')

# Perform a summary of the data
print('--------------------------------------')
print('Summary of data frame')
print('--------------------------------------')
print(df.describe())

# Correlation matrix
print('--------------------------------------')
print('Correlation Matrix')
print('--------------------------------------')
dfReadings = df.iloc[:,2:-1]
correlationMatrix = dfReadings.corr()
print(correlationMatrix)

# Visualization
# ---------------------------------------------
plt.style.use('ggplot')

# Distribution
# print('Plotting distribution charts...')
# print('--------------------------------------')
# dist_qn = input('Do you wish to plot all charts? (Y/N) ')
#
#
# if (dist_qn == 'Y') or (dist_qn == 'y'):
#     for i in range(2,56):
#         df.ix[:,i].plot

# Correlation Plot

def plot_correlation(dataframe, title=''):
    lang_names = dataframe.columns.tolist()
    tick_indices = np.arange(0.5, len(lang_names) + 0.5)
    plt.figure(figsize=(12,9))
    plt.pcolor(dataframe.values, cmap='RdBu', vmin=-1, vmax=1)
    colorbar = plt.colorbar()
    colorbar.set_label('Correlation')
    plt.title(title)
    plt.xticks(tick_indices, lang_names, rotation='vertical')
    plt.yticks(tick_indices, lang_names)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

corrPlot = plot_correlation(dfReadings.corr(),title='IMU readings')

#%%
# PCA
print('Run initial PCA on original DataFrame...')
pca = PCA()
dftr = pca.fit_transform(df.ix[:,2:42])
pca_comp = pca.components_
print('PCA Components:')
print(pca_comp)
print('--------------------------------------')
print('Find dominating columns...')
norm_col = []

for i in range(4):
    for j in range(len(dftr[i])):
        if (dftr[i][j] > 0.3) or (dftr[i][j] < -0.3):
            norm_col.append(j)

norm_col = list(set(norm_col))      # Take only unique column numbers

df_store = []   

for l in range(len(norm_col)):
        df_store.append(df.ix[:,norm_col[l]+2:(norm_col[l]+3)])

for dframe in range(len(df_store)):
    df_store[dframe].insert(0, 'index',list(range(len(df_store[dframe]))))

df_norm = functools.reduce(lambda left,right: pd.merge(left,right,on='index'), df_store)

print('--------------------------------------')
print('Normalizing data...')
print('--------------------------------------')
# Quick fix for zero range records
divisor = df_norm.max() - df_norm.min()
divisor.replace(0.0,1,inplace=True)

df_norm = (df_norm - df_norm.mean()) / 1

left = df.ix[:,0:2]
right = df_norm.ix[:,1:]

subjectID_col = df.ix[:,43:44]

df_new = left.join(right)   # Join timestamp, activity_id with normalized values
df_new = df_new.join(subjectID_col)

#df_new = df_new.fillna(method='ffill')      # Fill missing data
#df_new = df_new.fillna(method='bfill')      # Fill missing data

print(df_new.describe())

print('Conducting PCA...')
print('--------------------------------------')
dftr = pca.fit_transform(df_new.ix[:,2:42])

evr = pca.explained_variance_ratio_

print('Calculating which principal component(s) to keep based on explained variance score')
print('--------------------------------------')
total_exp_var = 0
i = 0
pcomp = []
        
while total_exp_var < 0.9:          # Takes principal component that explains at least 90% of variance, then stops loop
    total_exp_var = total_exp_var + evr[i]
    i += 1
    pcomp.append('PC%d' % i)
    

print('Principal component(s) to keep: %s' % ','.join(pcomp))
print('Principal component(s) explains %.2f percent of total variance' % float(total_exp_var * 100))

pltcol = int((len(pcomp)/len(pcomp))-1)


def plot_scree(explainedVarianceRatio, colsNumber, title=''):
    pc = np.arange(colsNumber) + 1

    plt.figure(figsize=(12,9))
    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('% of Variance Explained')
    plt.plot(pc,explainedVarianceRatio)

screePlot = plot_scree(evr,len(evr), 'Scree Plot')

# Print end time
print('End Time : ',timer.getTime())

# Plotting the graphs

plt.show()
