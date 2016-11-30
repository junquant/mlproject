import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

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
df.ix[:,:31] = (df.ix[:,:31] - df.ix[:,:31].mean()) / (df.ix[:,:31].max() - df.ix[:,:31].min())

subj = df.ix[:,-2]
activity = df.ix[:,-1]
subj_activity = (100*subj) + activity
df = pd.concat([df,subj_activity],axis=1)
df.rename(columns={0:'activity_subj'}, inplace=True)

# split data set into test and train using stratification (for both subj and activity)
# ---------------------
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=2016)

readings = df.ix[:,:-3]

# stratify based on subj_activity
for train_index, test_index in strat_split.split(readings,subj_activity):
    df_train, df_test = df.ix[train_index], df.ix[test_index]
    print('Size of data set: ', len(df))
    print('Size of training data set: ', len(train_index))
    print('Size of test data set: ', len(test_index))

print('Verifying distribution ...')
train_table = df_train.rename(index=str, columns={'subject':'training_count'})
test_table = df_test.rename(index=str, columns={'subject':'test_count'})
verify = pd.concat([train_table.ix[:,-2:].groupby('activity_subj').count(),
                    test_table.ix[:,-2:].groupby('activity_subj').count()],axis = 1)

pd.options.display.max_rows = 150

print(verify)

# ---------------------
# Subject and then Activity
# ---------------------
all_results = [] # to store all method results
method_results = {} # to store individual method results

# step 1.1 - get the readings data (from data stratified using activity)
readings_train = df_train.ix[:,:-3]
subj_train = df_train.ix[:,-3]
activity_train = df_train.ix[:,-2]
subj_activity_train = pd.DataFrame({'subject': subj_train, 'activity_id': activity_train})

# step 1.2 - fit the model to predict subject
print('Fitting model to predict subject ...')
clf = GaussianNB()
clf_multi = MultiOutputClassifier(clf)
time_bgn = time.time()
clf_multi.fit(readings_train, subj_activity_train)
dur_train_both = time.time() - time_bgn
predicted_subj_activity_train = clf_multi.predict(readings_train)
predicted_subj_activity_train = pd.DataFrame({'subject': predicted_subj_activity_train[:,1],
                                        'activity_id': predicted_subj_activity_train[:,0]})
predicted_subj = predicted_subj_activity_train.ix[:,1]
predicted_activity = predicted_subj_activity_train.ix[:,0]
predicted_subj_activity_train = (100*predicted_subj) + predicted_activity

# step 2.1 - get the readings data (from data stratified using subject)
readings_test = df_test.ix[:,:-3]
subj_test = df_test.ix[:,-3]
activity_test = df_test.ix[:,-2]
subj_activity_test = pd.DataFrame({'subject': subj_test, 'activity_id': activity_test})

# step 2.2 - predict subject activity
print('Predicting subject activity ... ')
predicted_subj_activity = clf_multi.predict(readings_test)
predicted_subj_activity = pd.DataFrame({'subject': predicted_subj_activity[:,1],
                                        'activity_id': predicted_subj_activity[:,0]})
predicted_subj = predicted_subj_activity.ix[:,1]
predicted_activity = predicted_subj_activity.ix[:,0]
predicted_subj_activity_test = (100*predicted_subj) + predicted_activity

# step 3 - printing results
actual_subj = df_test.ix[:,-3]
actual_activity = df_test.ix[:,-2]
subj_activity_test = (100*actual_subj) + actual_activity
subj_activity_train = (100*subj_train) + activity_train

ResultsWriter.write_to_file('results_junquan.txt',model='gnb_multioutput',
                            y_train_actual=subj_activity_train, y_train_predicted=predicted_subj_activity_train,
                            y_test_actual=subj_activity_test,y_test_predicted=predicted_subj_activity_test,
                            dur_train_activity=0, dur_train_subj=0, dur_train_both=dur_train_both,
                            method='both') # method = both / as / sa