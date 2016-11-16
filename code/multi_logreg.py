from datetime import datetime
import pandas as pd
from methods import MethodClass
from split import strat_split

target_variable = 'subject' # Define target variable

importData = strat_split(target_variable)  # Split data

# Load data
print('Loading data...')
start_time = datetime.now()
train_data = importData[0]
test_data = importData[1]
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
method = MethodClass()  # Instantiate MethodClass()
target_variables = method.target_variables  #  Load variables ['subject', 'activity_id']
A = target_variables[0]   # Set A as 'subject'
data = method.init_class(train_data, A)
print('Multinomial Logistic Regressions over c = 0.01, 0.1, 1, 10, 100')
print('--------------------------------------------------------------------')
print('Execute GridSearchCV with cv=10...')
parameters = [{
    'multi_class': ['multinomial'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg'],
    'max_iter': [100, 1000, 2500, 5000, 10000],
    }]
lr_best = GridSearchCV(LogisticRegression(), parameters, cv=10)
lr_best.fit(data[0], data[1])  # data[0] = x_train, data[1] = y_train
print('GridSearchCV Run #1 complete.')
print('--------------------------------------------------------------------')
print('The best parameters selected: ', lr_best.best_params_)
print('Best score produced with parameters: ', lr_best.best_score_)
print('--------------------------------------------------------------------')
print('Predict ', A)
# Instantiate lr with best parameters
lr = LogisticRegression(multi_class='multinomial',
                        solve='newton-cg',
                        c=lr_best.best_params_['C'])
lr.fit(data[0], data[1])  # Fit x_train, y_train
x_test = test_data.drop(['subject','activity_id'], axis=1)
print('Predicted classes for %s: ' % A)
y_pred = lr.predict(x_test)  # Predict using x_test
second_data = method.second_class(test_data, y_pred)
print('--------------------------------------------------------------------')
print('Running second classifier...')
lr2_best = GridSearchCV(LogisticRegression(), parameters, cv=10)
lr2_best.fit(second_data[0], second_data[1]) # second_data[0] = x2_train, second_data[1] = y2_train
end_time = datetime.now()
duration = end_time - start_time
print('GridSearchCV Run #2 complete.')
print('--------------------------------------------------------------------')
print('The best parameters selected: ', lr2_best.best_params_)
print('Best score produced with parameters: ', lr2_best.best_score_)
print('--------------------------------------------------------------------')
# TODO: Compute score for A + B
# Predicted correct over total number of observations
print('Time taken :', duration)
print('Task complete.')