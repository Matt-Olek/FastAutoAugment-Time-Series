# Defines the data loader for the baseline and augmented models using the UCRArchive_2018 dataset
import torch
import numpy as np
import pandas as pd

from FastAA.preprocess import preprocess_function

def getDataLoader(dataset_name, batch_size):
    
    # Load the dataset
    path = 'data/UCRArchive_2018/{}/'.format(dataset_name)
    
    train_file = path + '{}_TRAIN.tsv'.format(dataset_name)
    test_file = path + '{}_TEST.tsv'.format(dataset_name)
    
    # Open the files
    train_data = pd.read_csv(train_file, sep='\t', header=None)
    test_data = pd.read_csv(test_file, sep='\t', header=None)
    
    # Preprocess the data
    preprocess_function = preprocess_function(dataset_name)
    
    if preprocess_function is not None:
        f = lambda x: preprocess_function(x)
        train_data[0] = train_data[0].apply(lambda x: f(x))
        test_data[0] = test_data[0].apply(lambda x: f(x))
        
    train_np = train_data.to_numpy()
    test_np = train_data.to_numpy()   
    train = train_np.reshape(np.shape(train_np)[0], 1, np.shape(train_np)[1])
    test = test_np.reshape(np.shape(test_np)[0], 1, np.shape(test_np)[1])
    y_train = train[:, 0, 0]
    y_test = test[:, 0, 0]
    X_train = train[:, 0, 1:]
    X_test = test[:, 0, 1:]
    # To tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train, dtype=torch.int64).unsqueeze(1) 
    y_test = torch.tensor(y_test, dtype=torch.int64).unsqueeze(1) 
    
    train_dataset =[]
    test_dataset = []
    for i in range(len(X_train)):
        train_dataset.append((X_train[i], int(y_train[i].item())))
    for i in range(len(X_test)):
        test_dataset.append((X_test[i], int(y_test[i].item())))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader