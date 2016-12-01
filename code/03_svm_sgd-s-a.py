import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier

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
# Subject and then Activity
# ---------------------
for i in range(100):
    # step 1.1 - get the readings data (from data stratified using activity)
    readings_train = df_train_a.ix[:,:-2]
    subj_train = df_train_a.ix[:,-2]

    # step 1.2 - fit the model to predict subject
    print('Fitting model to predict subject ...')
    clf_subject = SGDClassifier(alpha=0.1)
    time_bgn = time.time()
    clf_subject.fit(readings_train, subj_train)
    dur_train_subj = time.time() - time_bgn

    # step 2.1 - get the readings data with subject
    readings_train = df_train_s.ix[:,:-2]
    activity_train = df_train_s.ix[:,-1]
    subj_train = df_train_s.ix[:,-2]

    # step 2.2 - join subject for training and fit the model to predict activity
    print('Fitting model to predict activity ...')
    readings_subj_train = np.column_stack((readings_train, subj_train))
    clf_activity = SGDClassifier(alpha=0.1)
    time_bgn = time.time()
    clf_activity.fit(readings_subj_train, activity_train)
    dur_train_activity = time.time() - time_bgn

    subj_train_for_result = subj_train
    activity_train_for_result = activity_train
    predicted_subj_train = clf_subject.predict(readings_train)
    predicted_activity_train = clf_activity.predict(readings_subj_train)


    # step 3.1 - predict subject and join it to the readings data
    print('Predicting subject ... ')
    readings_test = df_test_s.ix[:,:-2]
    predicted_subject = clf_subject.predict(readings_test)
    readings_subject_test = np.column_stack((readings_test,predicted_subject))

    # step 3.2 - predict subject
    print('Predicting activity ... ')
    predicted_activity = clf_activity.predict(readings_subject_test)

    # step 4 - printing results
    actual_activity_test = df_test_s.ix[:,-1]
    actual_subject_test = df_test_s.ix[:,-2]
    subj_activity_test = np.array((100*actual_subject_test) + actual_activity_test)
    predicted_subj_activity_test = (100*predicted_subject) + predicted_activity

    subj_activity_train = (100*subj_train_for_result) + activity_train_for_result
    predicted_subj_activity_train = (100*predicted_subj_train) + predicted_activity_train

    ResultsWriter.write_to_file('results_junquan_mbp.txt',model='svm_sgd_' + str(i+1),
                                y_train_actual=subj_activity_train, y_train_predicted=predicted_subj_activity_train,
                                y_test_actual=subj_activity_test,y_test_predicted=predicted_subj_activity_test,
                                dur_train_activity=dur_train_activity, dur_train_subj=dur_train_subj, dur_train_both=0,
                                method='sa') # method = both / as / sa