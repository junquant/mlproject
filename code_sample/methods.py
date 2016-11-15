# Define Errors
class Error(Exception):
    '''Base class for exceptions'''
    pass


class InputError(Error):
    '''Exception raised for errors in target argument'''
    def __init__(self, message):
        self.message = message


# Define methodology class
class MethodClass(object):
    
    def __init__(self):
        pass
    
    
    def init_class(self, train_data, target):
        '''
        Method for choosing first target variable.
        A -> B
        where A is subject and B is activity_id 
        or vice-versa.
        '''   
        target_variables = ['subject', 'activity_id']
        
        if target in target_variables:
            x_train = train_data.drop(target_variables, axis=1)  # Exclude 'activity_id', 'subject' from input variables
            y_train = train_data[target]  # assign training target
            return [x_train, y_train]
        else:
            message = 'Invalid target variable'
            raise InputError(message)