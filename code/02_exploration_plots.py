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

# Correlation Matrix
# ---------------------------------------------
plt.style.use('ggplot')

print('Plotting correlation ...')
plotdata = readings.corr()
lang_names = plotdata.columns.tolist()
tick_indices = np.arange(0.5, len(lang_names) + 0.5)

plt.figure(1,figsize=(10, 10))
plt.pcolor(plotdata.values, cmap='RdBu', vmin=-1, vmax=1)
colorbar = plt.colorbar()
colorbar.set_label('Correlation')
plt.title('IMU Readings')
plt.xticks(tick_indices, lang_names, rotation='vertical')
plt.yticks(tick_indices, lang_names)
plt.xlim(0,len(lang_names))
plt.ylim(0,len(lang_names))
plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

plt.savefig('../plots/correlation.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

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

# PCA Exploration
# ---------------------------------------------
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df_small_readings)

print('Plotting PCA 2 components ...')

pca = PCA()

dftr = pca.fit_transform(scaled_data)
dftr = np.column_stack((dftr[:,0:3],df_small_activity))

fig = plt.figure(2, figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.title('Top 2 Principal Components')
plt.scatter(dftr[:,0], dftr[:,1], c=dftr[:,3], alpha=0.5, cmap=plt.cm.prism)


plt.savefig('../plots/pca_2components.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Plotting PCA 3 components ...')

fig = plt.figure(3, figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('Top 3 Principal Components')
ax.scatter(dftr[:,0], dftr[:,1], dftr[:,2], c=dftr[:,3],marker='o', cmap=plt.cm.prism)

plt.savefig('../plots/pca_3components.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

print('Plotting Scree Plot ...')

scaled_data_full = minmax_scaler.fit_transform(readings)

pca_full = PCA()
pca_full.fit_transform(scaled_data_full)

comp = pca_full.components_
evr = pca_full.explained_variance_ratio_

pc = np.arange(len(comp)) + 1
fig = plt.figure(4, figsize=(12, 9))
fig.suptitle('Scree Plot')
ax = fig.add_subplot(111)
ax.set_xlabel('Principal Component')
ax.set_ylabel('% of Variance Explained')
ax.plot(pc, evr)

plt.savefig('../plots/screeplot.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# Variable Exploration
# ---------------------------------------------
print('Plotting Distribution ...')

fig = plt.figure(5, figsize=(14,14))
fig.suptitle('Histogram of all readings')
for i in range(0,len(df_small_readings.columns)):
    ax = fig.add_subplot(6, 6, i+1)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.hist(df_small_readings.iloc[:, i], bins=15 , color='#ffd700')
    ax.annotate(df_small_readings.iloc[:, i].name, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=8)

plt.figure(5)
plt.savefig('../plots/var_distribution.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# Density for each activity
# ---------------------------------------------
print('Plotting Density of each activity...')

# Get each unique activity and assign colours from the cmap
uniques = np.unique(df_small_activity)
numcolors = len(uniques)
cm = plt.get_cmap('terrain')
colors = []

for i in range(numcolors):
    colors.append(cm(1. * i / numcolors))

# Create the figure
fig = plt.figure(6, figsize=(12, 10))
fig.suptitle('Density of All Activities')
fig.text(x=0.1, y=0.5, s='Density of activities', rotation='vertical')

# Create the gaussian kernel
kde = KernelDensity(kernel='gaussian', bandwidth=10)

# For each column, tranform using gaussian kernel and plot the density
for i in range(0, len(df_small_readings.columns)):
    main_hist_data = df_small_readings.ix[:, i]

    # Calculate the density
    kde.fit(main_hist_data[:, np.newaxis])
    pdf_data = np.linspace(min(main_hist_data) - abs(min(main_hist_data) * 2),
                           max(main_hist_data) + abs(max(main_hist_data) * 2), 100)
    pdf = np.exp(kde.score_samples(pdf_data[:, np.newaxis]))

    # Annotate the plt and format it
    ax = fig.add_subplot(7, 5, i + 1)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.annotate(df_small_readings.iloc[:, i].name, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)

    # Plot the main density
    ax.plot(pdf_data, pdf, lw=1, c='red', fillstyle='bottom')
    ax.fill_between(pdf_data, pdf, alpha=.1, zorder=5, antialiased=True, color="#E01B6A")

    # Plot the density for each activity
    for j in range(0, len(uniques)):
        stacked = np.column_stack((main_hist_data, df_small_activity))
        activity_hist_data = stacked[stacked[:, 1] == uniques[j], 0]

        # Calculate the density for each activity
        kde.fit(activity_hist_data[:, np.newaxis])
        pdf_data = np.linspace(min(activity_hist_data) - abs(min(activity_hist_data) * 2),
                               max(activity_hist_data) + abs(max(activity_hist_data) * 2), 100)
        pdf = np.exp(kde.score_samples(pdf_data[:, np.newaxis]))

        # Plot the density
        ax.plot(pdf_data, pdf, lw=1, c=colors[j], label=uniques[j])

plt.figure(6)
plt.savefig('../plots/activities_distribution.png', format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)

# Density for each subject
# ---------------------------------------------
print('Plotting Density of each subject...')

# Get each unique activity and assign colours from the cmap
uniques = np.unique(df_small_subject)
numcolors = len(uniques)
cm = plt.get_cmap('terrain')
colors = []

for i in range(numcolors):
    colors.append(cm(1. * i / numcolors))

# Create the figure
fig = plt.figure(7, figsize=(12, 10))
fig.suptitle('Density of All Subjects')
fig.text(x=0.1, y=0.5, s='Density of subjects', rotation='vertical')

# Create the gaussian kernel
kde = KernelDensity(kernel='gaussian', bandwidth=10)

# For each column, tranform using gaussian kernel and plot the density
for i in range(0, len(df_small_readings.columns)):
    main_hist_data = df_small_readings.ix[:, i]

    # Calculate the density
    kde.fit(main_hist_data[:, np.newaxis])
    pdf_data = np.linspace(min(main_hist_data) - abs(min(main_hist_data) * 2),
                           max(main_hist_data) + abs(max(main_hist_data) * 2), 100)
    pdf = np.exp(kde.score_samples(pdf_data[:, np.newaxis]))

    # Annotate the plt and format it
    ax = fig.add_subplot(7, 5, i + 1)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.annotate(df_small_readings.iloc[:, i].name, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)

    # Plot the main density
    ax.plot(pdf_data, pdf, lw=1, c='red', fillstyle='bottom')
    ax.fill_between(pdf_data, pdf, alpha=.1, zorder=5, antialiased=True, color="#E01B6A")

    # Plot the density for each subject
    for j in range(0, len(uniques)):
        stacked = np.column_stack((main_hist_data, df_small_subject))
        subject_hist_data = stacked[stacked[:, 1] == uniques[j], 0]

        # Calculate the density for each subject
        kde.fit(subject_hist_data[:, np.newaxis])
        pdf_data = np.linspace(min(subject_hist_data) - abs(min(subject_hist_data) * 2),
                               max(subject_hist_data) + abs(max(subject_hist_data) * 2), 100)
        pdf = np.exp(kde.score_samples(pdf_data[:, np.newaxis]))

        # Plot the density
        ax.plot(pdf_data, pdf, lw=1, c=colors[j], label=uniques[j])

plt.figure(7)
plt.savefig('../plots/subject_distribution.png', format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)

print('End Time : ', timer.getTime())

# plt.show()
