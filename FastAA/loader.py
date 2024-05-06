import torch
import numpy as np
import pandas as pd
from transformations import apply_multiple_policies

def getDataLoader(dataset_name, batch_size, transform=None,num_opt=2):
    '''
    Defines the data loader for the baseline and augmented models using the UCRArchive_2018 dataset
    '''
    
    # Load the dataset
    path = 'data/UCRArchive_2018/{}/'.format(dataset_name)
    
    train_file = path + '{}_TRAIN.tsv'.format(dataset_name)
    test_file = path + '{}_TEST.tsv'.format(dataset_name)
    
    # Open the files
    train_data = pd.read_csv(train_file, sep='\t', header=None)
    test_data = pd.read_csv(test_file, sep='\t', header=None)
    
    # Get the number of classes
    nb_classes = len(train_data[0].unique())
    min_class = train_data[0].min()
    
    # Gets the classes from 0 to nb_classes
    if min_class != 0:
        train_data[0] = train_data[0] - min_class
        test_data[0] = test_data[0] - min_class
        
    print('Number of classes: {}'.format(nb_classes))
        
    train_np = train_data.to_numpy()
    test_np = test_data.to_numpy()
    train = train_np.reshape(np.shape(train_np)[0], 1, np.shape(train_np)[1])
    test = test_np.reshape(np.shape(test_np)[0], 1, np.shape(test_np)[1])
    
    y_train = train[:, 0, 0]
    y_test = test[:, 0, 0]
    X_train = train[:, 0, 1:]
    X_test = test[:, 0, 1:]
    if not transform==None:
        nb_batch_of_transform = len(transform)
        X_train_transformed = []
        print('Number of batch of transformations: {}'.format(nb_batch_of_transform))
        for batch in range(nb_batch_of_transform):
            nb_transforms = len(transform[batch])//(3*num_opt)
            print('Number of transformations in batch {}: {}'.format(batch, nb_transforms))
            X_train_transformed.append(apply_multiple_policies(X_train, transform[batch], num_opt))
        X_train = np.concatenate(X_train_transformed, axis=0)
        y_train = np.concatenate([y_train]*nb_batch_of_transform, axis=0)
        
            
            
            
        
    # To tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train, dtype=torch.int64).unsqueeze(1) 
    y_test = torch.tensor(y_test, dtype=torch.int64).unsqueeze(1) 
    
    # One-hot encoding
    y_train = torch.nn.functional.one_hot(y_train, num_classes=nb_classes)
    y_test = torch.nn.functional.one_hot(y_test, num_classes=nb_classes)
    
    train_dataset =[]
    for i in range(len(X_train)):
        train_dataset.append((X_train[i], y_train[i].squeeze(0)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = [X_test, y_test]
    return train_loader, test_dataset, nb_classes