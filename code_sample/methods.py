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
    
    def __init__(self, init_called=False):
        self.init_called = init_called
    
    
    def init_class(self, train_data, target, init_called):
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
            init_called = True  # set init_called to True for second_class function to run
            return [x_train, y_train]
        else:
            message = 'Invalid target variable'
            raise InputError(message)