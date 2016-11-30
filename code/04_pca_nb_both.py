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
data = np.loadtxt(filePath, delimiter = ',', skiprows = 1, dtype=dataType)
df = pd.DataFrame(data)

subj = df.ix[:,-2]
activity = df.ix[:,-1]
subj_activity = (100*subj) + activity
df = pd.concat([df,subj_activity],axis=1)
df.rename(columns={0:'activity_subj'}, inplace=True)

# split data set into test and train using stratification (for both subj and activity)
# ---------------------
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=2016)

# stratify based on subj_activity
for train_index, test_index in strat_split.split(df,subj_activity):
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
subj_activity_train = df_train.ix[:,-1]

# step 1.2 - scale to min 0 max 1 and Perform PCA
print('Performing PCA ...')
minmax_scaler = MinMaxScaler()
pca = PCA(n_components=10)

readings_train = minmax_scaler.fit_transform(readings_train)
readings_train = pca.fit_transform(readings_train)

# step 1.3 - fit the model to predict subject
print('Fitting model to predict subject ...')
clf_both = GaussianNB()
time_bgn = time.time()
clf_both.fit(readings_train, subj_activity_train)
dur_train_both = time.time() - time_bgn
predicted_subj_activity_train = clf_both.predict(readings_train)

# step 2.1 - get the readings data (from data stratified using subject)
readings_test = df_test.ix[:,:-3]

# step 2.2 - scale to min 0 max 1 and perform PCA
readings_test = minmax_scaler.fit_transform(readings_test)
readings_test = pca.fit_transform(readings_test)

# step 2.3 - predict subject activity
print('Predicting subject activity ... ')
predicted_subj_activity_test = clf_both.predict(readings_test)

# step 3 - printing results and writing to results to file
subj_activity_test = df_test.ix[:,-1]

ResultsWriter.write_to_file('results_junquan.txt',model='gnb_pca',
                            y_train_actual=subj_activity_train, y_train_predicted=predicted_subj_activity_train,
                            y_test_actual=subj_activity_test,y_test_predicted=predicted_subj_activity_test,
                            dur_train_activity=0, dur_train_subj=0, dur_train_both=dur_train_both,
                            method='both') # method = both / as / sa