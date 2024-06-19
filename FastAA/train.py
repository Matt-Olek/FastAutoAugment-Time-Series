# Defines the training process for the baseline and augmented models
import math
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from bayesian_optimization import bayesian_optimization
from sklearn.model_selection import StratifiedKFold
from loader import getDataLoader
from model import Classifier_RESNET
from config import device, is_WandB
from validation import validate
import wandb

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# ------------------------------------ Baseline Training ------------------------------------ #

def train_baseline(dataset_name, epochs,batch_size, runs):
    '''
    Defines the training process for the baseline models
    '''
    # Load the dataset
    train_loader, test_dataset,nb_classes = getDataLoader(dataset_name, batch_size)
    
    new_batch_size = int(len(train_loader.dataset)*0.01)
    if new_batch_size > batch_size:
        if new_batch_size > 128:
            new_batch_size = 128
        else:
            batch_size = new_batch_size
            print('Batch size updated to', batch_size)
            train_loader, test_dataset, nb_classes = getDataLoader(dataset_name, batch_size)
    
    # Accuracy, F1, Precision, Recall
    val_loss = []
    val_accuracy = []
    val_f1 = []
    val_recall = []
    # Run_loop
    for run in range(runs):
        print('Run {}/{}'.format(run + 1, runs))
        val_log = train_baseline_run(train_loader, test_dataset, dataset_name, nb_classes, epochs, batch_size, run_id=run)
        val_loss.append(val_log['loss'])
        val_accuracy.append(val_log['acc'])
        print('Accuracy:', val_log['acc'])
        val_f1.append(val_log['f1'])
        val_recall.append(val_log['recall'])
    
    # Compute the mean and standard deviation
    val_loss_mean = pd.Series(val_loss).mean()
    val_accuracy_mean = pd.Series(val_accuracy).mean()
    val_f1_mean = pd.Series(val_f1).mean()
    val_recall_mean = pd.Series(val_recall).mean()
    
    val_loss_std = pd.Series(val_loss).std()
    val_accuracy_std = pd.Series(val_accuracy).std()
    val_f1_std = pd.Series(val_f1).std()
    val_recall_std = pd.Series(val_recall).std()
    
    print('Mean Loss:', val_loss_mean, '±', val_loss_std)
    print('Mean Accuracy:', val_accuracy_mean, '±', val_accuracy_std)
    print('Mean F1:', val_f1_mean, '±', val_f1_std)
    print('Mean Recall:', val_recall_mean, '±', val_recall_std)
    
    # Write the results in a csv file
    results = pd.DataFrame({
        'dataset': [dataset_name],
        'type': ['baseline'],
        'loss_mean': [val_loss_mean],
        'loss_std': [val_loss_std],
        'accuracy_mean': [val_accuracy_mean],
        'accuracy_std': [val_accuracy_std],
        'f1_mean': [val_f1_mean],
        'f1_std': [val_f1_std],
        'recall_mean': [val_recall_mean],
        'recall_std': [val_recall_std],
        'nb_classes': nb_classes,
        'train_size': len(train_loader.dataset),
        'test_size': len(test_dataset[0])
    })
    
    # Check if the file already exists
    file_exists = os.path.isfile('data/logs/results.csv')
    
    # Write the results to the file
    results.to_csv('data/logs/results.csv', mode='a', header=not file_exists, index=False)
    
    if not file_exists:
        print('Results saved in data/logs/results.csv')
    else:
        print('Results appended to data/logs/results.csv')
        
        
        
def train_baseline_run(train_loader, test_dataset, dataset_name, nb_classes, epochs, batch_size, lr=0.001, weight_decay=0.0001,patience=100, run_id=0):
    '''
    Defines the training process for a single baseline model run
    '''
    # WandB - Initialize a new run
    if is_WandB:
        wandb.init(
            project='FastAA_full_datasets_exploration', 
            config={'type': 'baseline',  # 'baseline' or 'augmented
                    'dataset': dataset_name, 
                    'nb_classes': nb_classes, 
                    'epochs': epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay,
                    'run_id': run_id}
            )
    
    # Define the model
    model = Classifier_RESNET(input_shape=train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
    
    # Training
    best_acc = 0
    for epoch in tqdm(range(epochs)):
        epoch_avg_loss, epoch_accuracy = train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch)
        val_log = validate(test_dataset, model, criterion)

        # WandB - Log metrics
        if is_WandB:
            wandb.log({'epoch': epoch, 'train_loss': epoch_avg_loss, 'train_accuracy': epoch_accuracy, 'val_loss': val_log['loss'], 'val_accuracy': val_log['acc'], 'val_f1': val_log['f1']})

    # WandB - Finish the run
    if is_WandB:
        wandb.finish()
    
    return val_log
        
        
def train_epoch(train_loader, model, criterion, optimizer, scheduler,epoch_id):
    '''
    Training epoch
    '''
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device).float()
        target = target.to(device).float()
        output = model(input).float()
        loss = criterion(output, target)
        acc = torch.sum(torch.argmax(output, dim=1) == torch.argmax(target, dim=1))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
            
        train_loss += loss.item()
        correct += acc.item()
        total += target.size(0)
    epoch_avg_loss = train_loss / total
    epoch_accuracy = correct / total
    return epoch_avg_loss, epoch_accuracy
    
# ------------------------------------ FastAA Training ------------------------------------ #
            
def train_fastAA(dataset_name, epochs, batch_size, K, N, T, B, runs, lr=0.0001, weight_decay=0.0001, patience=100):
    # Load the dataset
    train_loader, test_dataset,nb_classes = getDataLoader(dataset_name, batch_size=None)
    
    new_batch_size = int(len(train_loader.dataset)*0.1) 
    if new_batch_size > batch_size :
        batch_size = new_batch_size
        print('Batch size updated to', batch_size)
        
    # Accuracy, F1, Precision, Recall
    val_loss = []
    val_accuracy = []
    val_f1 = []
    val_recall = []
    # Run_loop
    for run in range(runs):
        print('Run {}/{}'.format(run + 1, runs))
        val_log = train_fastAA_run(dataset_name, epochs, batch_size, K, N, T, B, lr, weight_decay, patience)
        val_loss.append(val_log['loss'])
        val_accuracy.append(val_log['acc'])
        print('Accuracy:', val_log['acc'])
        val_f1.append(val_log['f1'])
        val_recall.append(val_log['recall'])
        
    val_loss_mean = pd.Series(val_loss).mean()
    val_accuracy_mean = pd.Series(val_accuracy).mean()
    val_f1_mean = pd.Series(val_f1).mean()
    val_recall_mean = pd.Series(val_recall).mean()
    
    val_loss_std = pd.Series(val_loss).std()
    val_accuracy_std = pd.Series(val_accuracy).std()
    val_f1_std = pd.Series(val_f1).std()
    val_recall_std = pd.Series(val_recall).std()
    
    print('Mean Loss:', val_loss_mean, '±', val_loss_std)
    print('Mean Accuracy:', val_accuracy_mean, '±', val_accuracy_std)
    print('Mean F1:', val_f1_mean, '±', val_f1_std)
    print('Mean Recall:', val_recall_mean, '±', val_recall_std)
    
        
    # Write the results in a csv file
    results = pd.DataFrame({
        'dataset': [dataset_name],
        'type': ['augmented'],
        'loss_mean': [val_loss_mean],
        'loss_std': [val_loss_std],
        'accuracy_mean': [val_accuracy_mean],
        'accuracy_std': [val_accuracy_std],
        'f1_mean': [val_f1_mean],
        'f1_std': [val_f1_std],
        'recall_mean': [val_recall_mean],
        'recall_std': [val_recall_std],
        'nb_classes': nb_classes,
        'train_size': len(train_loader.dataset),
        'test_size': len(test_dataset[0])
    })
    
    results.to_csv('data/logs/results.csv', mode='a', header=False, index=False)

def train_fastAA_run(dataset_name, epochs, batch_size, K, N, T, B, lr=0.001, weight_decay=0.0001, patience=100):
    '''
    Defines the training process for the FastAA augmented models
    '''
    train_loader, test_dataset, nb_classes = getDataLoader(dataset_name, batch_size=None)    # Load the dataset with personalized loader (#loader.py)
    
    # ----------------------------------- K-Fold Bagging ----------------------------------- #
    
    X,y = zip(*train_loader.dataset)
    y = np.argmax(y, axis=1)
    
    kf = StratifiedKFold(n_splits=K, shuffle=True)
    
    train_folds = []
    for fold, (train_index, _) in enumerate(kf.split(X, y)):
        train_folds.append(torch.utils.data.Subset(train_loader.dataset, train_index))
    
    # ------------------------------------ Training Child Models and Bayesian Optimization ------------------------------------ #
    
    T_star = []
    
    for fold in range(K):
        print('Fold {}/{}'.format(fold + 1, K))
        # Define the child model
        model = Classifier_RESNET(input_shape=train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
        
        # Split fold into two parts
        fold_len = len(train_folds[fold])
        train_fold, bayes_fold = torch.utils.data.random_split(train_folds[fold], [int(fold_len * 0.5), fold_len - int((fold_len * 0.5))])
        
        # Training
        child_batch_size = int(batch_size/(K))
        child_batch_size = max(8, child_batch_size)
        train_loader = torch.utils.data.DataLoader(train_fold, batch_size=child_batch_size, shuffle=True)
        best_acc = 0
        model.train()
        for epoch in tqdm(range(epochs)):
            epoch_avg_loss, epoch_accuracy = train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch)
            val_log = validate(test_dataset, model, criterion)
            
            # # Save the best model
            # if val_log['acc'] > best_acc:
            #     best_acc = val_log['acc']
            # torch.save(model.state_dict(), 'model_fold{}.pth'.format(fold))
        
        # # Load the best model
        # model.load_state_dict(torch.load('model_fold{}.pth'.format(fold)))
        model.eval()
        
        # Perform bayesian optimization using the other part of the fold
        T_star.append(bayesian_optimization(model, bayes_fold, N, T, B, criterion, num_opt=2))
    
    print(len(T_star), "total policies found")
    print(T_star)
    
    # ------------------------------------ Validation Initialization ------------------------------------ #
    
    if is_WandB:
        wandb.init(
            project='FastAA_full_datasets_exploration', 
            config={'type': 'augmented',  # 'baseline' or 'augmented
                    'dataset': dataset_name, 
                    'nb_classes': nb_classes, 
                    'epochs': epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay,
                    'K': K,
                    'N': N,
                    'T': T,
                    'B': B}
            )
    
    augmented_batch_size = batch_size #* (1 + len(T_star))
    augmented_train_loader, _, _ = getDataLoader(dataset_name, augmented_batch_size, transform=T_star, num_opt=2)
    
    model_augmented = Classifier_RESNET(input_shape=augmented_train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_augmented.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
    
    # ------------------------------------ Training ------------------------------------ #
    
    best_acc = 0
    
    for epoch in tqdm(range(epochs)):
        epoch_avg_loss, epoch_accuracy = train_epoch(augmented_train_loader, model_augmented, criterion, optimizer, scheduler, epoch)
        val_log = validate(test_dataset, model_augmented, criterion)
        
        # WandB - Log metrics
        if is_WandB:
            wandb.log({'epoch': epoch, 'train_loss': epoch_avg_loss, 'train_accuracy': epoch_accuracy, 'val_loss': val_log['loss'], 'val_accuracy': val_log['acc'], 'val_f1': val_log['f1']})
        
        # Save the best model
        if val_log['acc'] > best_acc:
            best_acc = val_log['acc']
            torch.save(model_augmented.state_dict(), f'models/best_model_augmented_{dataset_name}.pth')
        
        scheduler.step(val_log['loss'])
        
    # ------------------------------------ Finish ------------------------------------ #
    
    if is_WandB:
        wandb.finish()
        
    return val_log
        


        
    
    

        
        
        
        
    
    

    