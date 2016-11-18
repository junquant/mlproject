from datetime import datetime
import numpy as np
import pandas as pd
from methods import InputError


def strat_split(target_variable):
    if target_variable in ['subject', 'activity_id']:
        # Load data
        start_time = datetime.now()
        data = pd.read_csv('../data/consolidated_clean_all.txt', sep=',')
        # data = pd.read_csv('../code/101.txt', sep=',')
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
                                             test_size=0.25,
                                             random_state=random_state)
        x = data.drop(target_variable,axis=1) # Set input variables (all except target_variable)
        y = data.ix[:, target_variable] # Set output variable
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
        print('Merging input and target columns...')
        # Merge input and target columns
        train_data = x_train.join(y_train)
        test_data = x_test.join(y_test)
        print('Data saved in list [train_data, test_data]')
        return [train_data, test_data]
    else:
        message = 'Invalid target variable.'
        raise InputError(message)