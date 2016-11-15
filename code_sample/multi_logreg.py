from datetime import datetime
import pandas as pd
from methods import MethodClass

# Load data
print('Loading data via "train_data.txt", "test_data.txt"...')
start_time = datetime.now()
train_data = pd.read_csv('../data/train_data.txt', sep=',')
test_data = pd.read_csv('../data/test_data.txt', sep=',')
end_time = datetime.now()
duration = end_time - start_time
print('Date files loaded.')
print('Time taken: ', duration)
print('--------------------------------------------------------------------')

# Multinomial Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

print('Preparing data...')
start_time = datetime.now()
method = MethodClass()
init_target = 'subject'
data = method.init_class(train_data, init_target)
print('Multinomial Logistic Regressions over c = 0.01, 0.1, 1, 10, 100')
print('--------------------------------------------------------------------')
print('Execute GridSearchCV with cv=10...')
parameters = [{
    'multi_class': ['multinomial'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg'],
    'max_iter': [10000],
    }]
lr_best = GridSearchCV(LogisticRegression(), parameters, cv=10)
lr_best.fit(data[0], data[1])  # data[0] = x_train, data[1] = y_train
print('GridSearchCV complete.')
print('--------------------------------------------------------------------')
print('The best parameters selected: ', lr_best.best_params_)
print('Best score produced with parameters: ', lr_best.best_score_)
print('--------------------------------------------------------------------')
print('Predict ', init_target)
lr = LogisticRegression(multi_class='multinomial',
                        solve='newton-cg',
                        c=lr_best.best_params_['C'])
lr.fit(data[0], data[1])  # Fit x_train, y_train
x_test = test_data.drop(['subject','activity_id'], axis=1)
print('Predicted classes for %s: ' % init_target)
y_pred = lr.predict(x_test)  # Predict using x_test
# Join predicted classes to test data - becomes train_data for next run
second_target = 'activity_id'
x2_test = x_test.join(y_pred)
y2_test = test_data['activity_id']  # Set 'activity_id' as target variable
print('--------------------------------------------------------------------')
print('Running second classifier...')
lr2_best = GridSearchCV(LogisticRegression(), parameters, cv=10)
lr2_best.fit(x2_test, y2_test)
end_time = datetime.now()
duration = end_time - start_time
print('Best score produced: ', lr2_best.best_score_)
print('Time taken :', duration)
print('Task complete.')