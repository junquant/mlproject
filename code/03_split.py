from datetime import datetime
import numpy as np
import pandas as pd


# Load data
print('Loading data via "consolidated_clean.txt"...')
start_time = datetime.now()
data = pd.read_csv('../data/consolidated_clean.txt', sep=',')
end_time = datetime.now()
duration = end_time - start_time
print('Date file loaded.')
print('Time taken: ', duration)
print('--------------------------------------------------------------------')


# Stratified Shuffle Split
print('Stratified splitting in progress...')
start_time = datetime.now()

from sklearn.model_selection import StratifiedShuffleSplit

random_state = np.random.seed(2016) # Set random seed

# Instantiate Stratified Shuffle Split
strat_split = StratifiedShuffleSplit(n_splits=1,
                                     train_size=0.75,
                                     random_state=random_state)
x = data.ix[:,:42] # Set input variables (all except 'subject')
x = x.drop('activity_id', axis=1) # Drop 'activity_id' from input variables
y = data.ix[:, 42] # Set output variable to be 'subject'

for train_index, test_index in strat_split.split(x,y):
    print('TRAIN indices: ', train_index)
    print('TEST indices: ', test_index)
    x_train, x_test = x.ix[train_index], x.ix[test_index]
    y_train, y_test = y.ix[train_index], y.ix[test_index]

end_time = datetime.now()
duration = end_time - start_time
print('Train and test data stratified splitting complete.')
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)
print('Time taken: ', duration)
print('--------------------------------------------------------------------')

# Save output files - train_data.txt, test_data.txt
print('Saving files...')
# Merge input and target columns
train_data = x_train.join(y_train)
test_data = x_test.join(y_test)
train_data.to_csv('../data/train_data.txt')
test_data.to_csv('../data/test_data.txt')
print('Files saved as "train_data.txt" & "test_data.txt"')