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

# SVM
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
print('Preparing data...')
start_time = datetime.now()
method = MethodClass()  # Instantiate MethodClass()
target_variables = method.target_variables  #  Load variables ['subject', 'activity_id']
A = target_variables[0]   # Set A as 'subject'
data = method.init_class(train_data, A)
print('LinearSVC over c = 0.01, 0.1, 1, 10, 100...')
print('--------------------------------------------------------------------')
c = [0.01, 0.1, 1, 10, 100]
print('Execute GridSearchCV with cv=10...')
parameters = [
    {'multi_class':['ovr'], 'C': c}
]
svc_best = GridSearchCV(LinearSVC(), parameters, cv=10)
svc_best.fit(data[0], data[1])  # data[0] = x_train, data[1] = y_train
print('GridSearchCV Run #1 complete.')
print('--------------------------------------------------------------------')
print('The best parameters selected: ', svc_best.best_params_)
print('Best score produced with parameters: ', svc_best.best_score_)
print('--------------------------------------------------------------------')
print('Predict ', A)
# Instantiate lr with best parameters
best_c = svc_best.best_params_['C']
svc = LinearSVC(multi_class='ovr', C=best_c)
svc.fit(data[0], data[1])  # Fit x_train, y_train
x_test = test_data.drop(target_variables, axis=1)
y_test = test_data[A]
y_pred = svc.predict(x_test)  # Predict using x_test
print('Predicted classes for %s:' % A)
print(y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
print('--------------------------------------------------------------------')
print('Running second classifier...')
B = target_variables[1]
x2_train = train_data.drop(B, axis=1)
y2_train = train_data[B]
svc2_best = GridSearchCV(LinearSVC(), parameters, cv=10)
svc2_best.fit(x2_train, y2_train)
end_time = datetime.now()
duration = end_time - start_time
print('GridSearchCV Run #2 complete.')
print('--------------------------------------------------------------------')
print('The best parameters selected: ', svc2_best.best_params_)
print('Best score produced with parameters: ', svc2_best.best_score_)
print('--------------------------------------------------------------------')
best_c2 = svc2_best.best_params_['C']
svc2 = LinearSVC(multi_class='ovr', C=best_c2)
svc2.fit(x2_train, y2_train)  # Fit x2_train, y2_train
x2_test = test_data.drop(B, axis=1)
y2_test = test_data[B]
y2_pred = svc2.pred(x2_test)
print('Predicted classes for %s:' % B)
print(y2_pred)
accuracy2 = metrics.accuracy_score(y2_test, y2_pred)
print('Accuracy: ', accuracy2)
print('--------------------------------------------------------------------')
# Predict first run and use results to predict second run
x3_test = x_test.join(y_pred)
y3_pred = svc.predict(x3_test)
from sklearn.metrics import classification_report
import utilities
metadata = utilities.MetaData()
print('Computing final score...')
print('Results for Step #1 - predict A:')
print(classification_report(y_test, y_pred, target_names=metadata.activities))
print('Results for Step #2 - predict B, given predicted A:')
print(classification_report(y2_test, y3_pred), target_names=list(range(8)))
print('Time taken :', duration)
print('Task complete.')