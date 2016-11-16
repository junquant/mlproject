# Define Errors
class Error(Exception):
    '''Base class for exceptions'''
    pass


class InputError(Error):
    '''Exception raised for errors in target argument'''
    def __init__(self, message):
        self.message = message

class ProcessError(Error):
    '''Exeception raised when second_class is called before init_class'''
    def __init__(self, message):
        self.message = message


# Define methodology class
class MethodClass(object):
    
    def __init__(self):
        self.target_variables = ['subject', 'activity_id']
    
    def init_class(self, train_data, A):
        '''
        Method for choosing first target variable.
        A -> B
        where A is subject and B is activity_id 
        or vice-versa.
        
        Takes in two arguments: train_data, target
        
        Returns [x_train, y_train]
        '''
        target_variables = self.target_variables
        
        if A in target_variables:
            x_train = train_data.drop(target_variables, axis=1)  # Exclude 'activity_id', 'subject' from input variables
            y_train = train_data[A]  # assign training target
            target_variables.remove(A)
            return [x_train, y_train]
        else:
            message = 'Invalid target variable'
            raise InputError(message)
    
    def second_class(self, test_data, y_pred):
        '''
        Method for choosing second target variable.
        Joins predicted result to second training set.
        
        Takes in 3 arguments: test_data, target, y_pred
        
        Returns [x2_train, y2_train]
        '''
        target_variables = self.target_variables
        
        if ['subject', 'activity_id'] in test_data.columns:
            message = 'Target variables exist in input table'
            raise InputError(message)
        else:
            if len(target_variables) == 1:
                target = target_variables[0]
                x2_train = test_data.join(y_pred)
                y2_train = test_data[target]
                return [x2_train, y2_train]
            else:
                message = 'init_class() method has not been called yet.'
                raise ProcessError(message)