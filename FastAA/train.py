# Defines the training process for the baseline and augmented models

from FastAA.loader import getDataLoader
from FastAA.model import Classifier_RESNET

def train_baseline(dataset_name, nb_classes,epochs,batch_size):
    '''
    Defines the training process for the baseline and augmented models
    '''
    
    # Load the dataset
    train_loader, test_loader = getDataLoader(dataset_name, batch_size)
    
    # Define the model
    model 
    

    