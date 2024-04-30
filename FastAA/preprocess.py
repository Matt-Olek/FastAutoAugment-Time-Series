# Preprocess the UCR datasets to fit the FastAA model

def preprocess_function(dataset_name):
    if dataset_name == 'ECG200':
        def f(x):
            if x ==-1:
                x = 0
            return x
        return f
    elif dataset_name == 'ECG5000':
        def f(x):
            return x-1
        return f
    elif dataset_name == 'Worms':
        def f(x):
            return x-1
    else :
        return None