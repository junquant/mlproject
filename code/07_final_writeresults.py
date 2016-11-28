import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale

from utilities import Timer, MetaData, ResultsWriter

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated_clean_101.txt'
resultsPath = '../data/results.txt'

metadata = MetaData()
dataType = metadata.getProcessedColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
data = np.loadtxt(filePath, delimiter = ',', skiprows = 1, dtype=dataType)
df = pd.DataFrame(data)

subj = df.ix[:,-2]
activity = df.ix[:,-1]
subj_activity = (100*subj) + activity
df = pd.concat([df,subj_activity],axis=1)
df.rename(columns={0:'subj_activity'}, inplace=True)

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
verify = pd.concat([train_table.ix[:,-2:].groupby('subj_activity').count(),
                    test_table.ix[:,-2:].groupby('subj_activity').count()],axis = 1)

pd.options.display.max_rows = 150

print(verify)

# ---------------------
# Subject and then Activity
# ---------------------
# step 1.1 - get the readings data (from data stratified using activity)
readings_train = df_train.ix[:,:-3]
readings_train = scale(readings_train)
subj_activity_train = df_train.ix[:,-1]

# step 1.3 - fit the model to predict subject
print('Fitting model to predict subject ...')
clf_both = GaussianNB()
time_bgn = time.time()
clf_both.fit(readings_train, subj_activity_train)
dur_train_both = time.time() - time_bgn

# step 2.1 - get the readings data (from data stratified using subject)
readings_test = df_test.ix[:,:-3]
readings_test = scale(readings_test)

# step 2.3 - predict subject activity
print('Predicting subject activity ... ')
predicted_subj_activity = clf_both.predict(readings_test)

# step 3 - printing results and writing to results to file
df_test = df_test.reset_index()
predicted_subj_activity = pd.DataFrame(predicted_subj_activity)
predicted_subj_activity.rename(columns={0:'predicted_activity_subj'}, inplace=True)

result = pd.concat([df_test.ix[:,1:], predicted_subj_activity],axis=1)

result.to_csv(resultsPath, header=True, index=None, sep=',')

