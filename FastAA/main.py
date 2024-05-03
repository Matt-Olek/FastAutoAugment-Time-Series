# Implementation of hyperparameter selection and base commands for Fast AutoAugment

from train import train_baseline, train_fastAA
from config import device

if __name__ == "__main__":
    dataset_name = 'TwoLeadECG'
    
    # Hyperparameters - FastAA
    
    K=5                     # Number of folds
    N=10                    # Number of best policies
    T=2                     # Number of iterations
    B=200                   # Number of samples per fold
    
    # Hyperparameters - Baseline and comparison
    
    epochs = 200            # Number of epochs
    batch_size = 20       # Batch size
    comparison = True       # Compare FastAA with no augmentation
    
    print('\n#################################################\n')
    print('Performing Fast AutoAugment on the {} dataset'.format(dataset_name), 'computing on device {}'.format(device))
    print('Using {}-fold bagging (K)'.format(K), 'to find the N={} best policies'.format(N), 'over T={} iterations'.format(T), 'with B={} samples per fold'.format(B))
    print('\n#################################################\n')
    
    if comparison:
        print('Comparing FastAA results with baseline results')         # And random augmentation soon
        print('Training the baseline model on the {} dataset ...'.format(dataset_name))
        print('\n#################################################\n')
        # train_baseline(dataset_name, epochs, batch_size)
        print('\n#################################################\n')
        print('Baseline training completed')
        print('Training FastAA on the {} dataset ...'.format(dataset_name))
        print('\n#################################################\n')
        train_fastAA(dataset_name, K, N, T, B, epochs, batch_size)
        
        
        
        
    