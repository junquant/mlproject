import numpy as np
import pandas as pd
from datetime import datetime


# Load data
print('Loading data via "consolidated.txt"...')
start_time = datetime.now()
data = pd.read_csv('../data/consolidated.txt', sep=',')
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
x = data.ix[:,:53] # Set input variables
y = data.ix[:, 54] # Set output variable to be 'subject'

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


# Multinomial Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

c = [0.01, 0.1, 1, 10, 100]

for i in range(len(c)):
    lr = LogisticRegression(multi_class = 'multinomial',
                            solver = 'newton-cg',
                            C = c[i])    
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('At C: %.2f' % c[i])
    print('Accuracy of LR: ', accuracy)
    print('--------------------')