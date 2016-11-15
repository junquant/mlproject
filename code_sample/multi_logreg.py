from datetime import datetime
import pandas as pd


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

print('Multinomial Logistic Regressions over c = 0.01, 0.1, 1, 10, 100')
print('--------------------------------------------------------------------')
print('Execute GridSearchCV with cv=10...')
start_time = datetime.now()
c = [0.01, 0.1, 1, 10, 100]
parameters = [
    {'C': c}
]

lr_best = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='newton-cg'), parameters, cv=10)
end_time = datetime.now()
duration = end_time - start_time
print('GridSearchCV complete.')
print('Time taken: ', duration)
print('--------------------------------------------------------------------')
print('The best parameters selected: ', lr_best.best_params_)
print('Best score produced with parameters: ', lr_best.best_score_)
print('--------------------------------------------------------------------')