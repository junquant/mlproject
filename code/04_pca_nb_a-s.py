import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

from utilities import Timer, MetaData, ResultsWriter

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_101.txt'

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

subj = df.ix[:,-2]
activity = df.ix[:,-1]

# split data set into test and train using stratification (for both subj and activity)
# ---------------------
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=2016)

# stratify based on activity
for train_index, test_index in strat_split.split(df,activity):
    df_train_a, df_test_a = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of training data set: ', len(train_index))
    print('Size of test data set: ', len(test_index))

# stratify based on subject
for train_index, test_index in strat_split.split(df,subj):
    df_train_s, df_test_s = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of training data set: ', len(train_index))
    print('Size of test data set: ', len(test_index))

print('Verifying distribution ...')
train_table = df_train_a.rename(index=str, columns={'subject':'training_count'})
test_table = df_test_a.rename(index=str, columns={'subject':'test_count'})
verify = pd.concat([train_table.ix[:,-2:].groupby('activity_id').count(),
                    test_table.ix[:,-2:].groupby('activity_id').count()],axis = 1)
print(verify)

train_table = df_train_s.rename(index=str, columns={'activity':'training_count'})
test_table = df_test_s.rename(index=str, columns={'activity':'test_count'})
verify = pd.concat([train_table.ix[:,-2:].groupby('subject').count(),
                    test_table.ix[:,-2:].groupby('subject').count()],axis = 1)
print(verify)

# ---------------------
# Activity and then Subject
# ---------------------
all_results = [] # to store all method results
method_results = {} # to store individual method results

# step 1.1 - get the readings data (from data stratified using activity)
readings_train = df_train_a.ix[:,:-2]
activity_train = df_train_a.ix[:,-1]

# step 1.2 - scale to min 0 max 1 and Perform PCA
print('Performing PCA ...')
minmax_scaler = MinMaxScaler()
pca = PCA(n_components=10)

readings_train = minmax_scaler.fit_transform(readings_train)
readings_train = pca.fit_transform(readings_train)

# step 1.3 - fit the model to predict activity
print('Fitting model to predict activity ...')
clf_activity = GaussianNB()
time_bgn = time.time()
clf_activity.fit(readings_train, activity_train)
dur_train_activity = time.time() - time_bgn

# step 2.1 - get the readings data with activity
readings_train = df_train_s.ix[:,:-1]
activity_train = df_train_s.ix[:,-1]
subj_train = df_train_s.ix[:,-2]

# step 2.2 - scale to min 0 max 1 and Perform PCA
print('Performing PCA ...')
readings_train = minmax_scaler.fit_transform(readings_train)
readings_train = pca.fit_transform(readings_train)

# step 2.3 - join activity for training and fit the model to predict subject
print('Fitting model to predict subject ...')
readings_act_train = np.column_stack((readings_train, activity_train))
clf_subject = GaussianNB()
time_bgn = time.time()
clf_subject.fit(readings_act_train, subj_train)
dur_train_subj = time.time() - time_bgn

# step 3.1 - get the readings data (from data stratified using subject)
readings_test = df_test_s.ix[:,:-2]

# step 3.2 - scale to min 0 max 1 and perform PCA
readings_test = minmax_scaler.fit_transform(readings_test)
readings_test = pca.fit_transform(readings_test)

# step 3.3 - predict activity and join it to the readings data
print('Predicting activity ... ')
predicted_activity = clf_activity.predict(readings_test)
readings_activity_test = np.column_stack((readings_test,predicted_activity))

# step 3.4 - predict subject
print('Predicting subject ... ')
predicted_subject = clf_subject.predict(readings_activity_test)
print(predicted_subject)
# step 3.3 - join predicted activity to the
actual_activity_test = df_test_s.ix[:,-1]
actual_subject_test = df_test_s.ix[:,-2]

# step 4 - printing results
actual_subj_activity = np.array((100*actual_subject_test) + actual_activity_test)
predicted_subj_activity = (100*predicted_subject) + predicted_activity

ResultsWriter.write_to_file('results_junquan.txt',model='pca_gnb',
                            y_actual=actual_subj_activity,y_predicted=predicted_subj_activity,
                            dur_train_activity=dur_train_activity, dur_train_subj=dur_train_subj,
                            dur_train_both=0,
                            method='as') # method = both / as / sa