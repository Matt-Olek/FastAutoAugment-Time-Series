# Implementation of hyperparameter selection and base commands for Fast AutoAugment

from FastAA.train import train_baseline

if __name__ == "__main__":
    dataset_name = 'ECG5000'
    nb_classes = 5
    
    # Hyperparameters - FastAA
    
    K=5                     # Number of folds
    N=10                    # Number of best policies
    T=2                     # Number of iterations
    B=200                   # Number of samples per fold
    
    # Hyperparameters - Baseline and comparison
    
    epochs = 100            # Number of epochs
    batch_size = 64         # Batch size
    comparison = True       # Compare FastAA with no augmentation
    
    print ('\n#################################################\n')
    print ('Performing Fast AutoAugment on the {} dataset'.format(dataset_name))
    print ('Number of classes: {}'.format(nb_classes))
    print ('Using {}-fold bagging (K)'.format(K), 'to find the N={} best policies'.format(N), 'over T={} iterations'.format(T), 'with B={} samples per fold'.format(B))
    print ('\n#################################################\n')
    
    if comparison:
        print ('Comparing FastAA results with no augmentation') # And random augmentation soon
        print ('\n#################################################\n')
        
        train_baseline(dataset_name, nb_classes, epochs, batch_size)
        
    # Load the dataset
    