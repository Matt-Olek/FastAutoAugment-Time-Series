# Defines the training process for the baseline and augmented models
from bayesian_optimization import bayesian_optimization
from loader import getDataLoader
from model import Classifier_RESNET
from config import device, is_WandB
from validation import validate
from torch import nn, optim
from torch.optim import lr_scheduler
import pandas as pd
import wandb
from tqdm import tqdm
import math
import torch

def train_baseline(dataset_name, epochs,batch_size,lr=0.001, weight_decay=0.0001,patience=100):
    '''
    Defines the training process for the baseline and augmented models
    '''
    # Load the dataset
    train_loader, test_dataset,nb_classes = getDataLoader(dataset_name, batch_size)
    
    # WandB - Initialize a new run
    if is_WandB:
        wandb.init(
            project='FastAA', 
            config={'type': 'baseline',  # 'baseline' or 'augmented
                    'dataset': dataset_name, 
                    'nb_classes': nb_classes, 
                    'epochs': epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay}
            )
    
    # Define the model
    model = Classifier_RESNET(input_shape=train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
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
        
        # Save the best model
        if val_log['acc'] > best_acc:
            best_acc = val_log['acc']
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step(val_log['loss'])
        
    # WandB - Finish the run
    if is_WandB:
        wandb.finish()
        
        
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
            
def train_fastAA(dataset_name, K, N, T, B, epochs, batch_size, lr=0.001, weight_decay=0.0001, patience=100):
    '''
    Defines the training process for the FastAA augmented models
    '''
    # Load the dataset
    train_loader, test_dataset,nb_classes = getDataLoader(dataset_name, batch_size=None)
    
    # WandB - Initialize a new run
    if is_WandB:
        wandb.init(
            project='FastAA', 
            config={'type': 'augmented',  # 'baseline' or 'augmented
                    'dataset': dataset_name, 
                    'nb_classes': nb_classes, 
                    'epochs': epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay}
            )
    
    # Splits the training set into K folds
    fold_size = math.ceil(len(train_loader.dataset) / K)
    lengths = [fold_size] * K
    if sum(lengths) > len(train_loader.dataset):
        lengths[-1] = len(train_loader.dataset) - sum(lengths[:-1])
    train_folds = torch.utils.data.random_split(train_loader.dataset, lengths)
    
    T_star=[]
    
    for fold in range(K):
        print('Fold {}/{}'.format(fold+1, K))
        # Define the child model
        model = Classifier_RESNET(input_shape=train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
        
        # Training
        child_batch_size = int(batch_size/K)
        train_loader = torch.utils.data.DataLoader(train_folds[fold], batch_size=child_batch_size, shuffle=True)
        best_acc = 0
        # for epoch in tqdm(range(epochs)):
        #     epoch_avg_loss, epoch_accuracy = train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch)
        #     val_log = validate(test_dataset, model, criterion)
        #     scheduler.step(val_log['loss'])
            
        #     # Save the best model
        #     if val_log['acc'] > best_acc:
        #         best_acc = val_log['acc']
        #         torch.save(model.state_dict(), 'best_model_fold{}.pth'.format(fold))
        
        # Load the best model
        model.load_state_dict(torch.load('best_model_fold{}.pth'.format(fold)))
        model.eval()
        
        # Perform bayesian optimization to find the best policies
        T_star.append(bayesian_optimization(model, train_folds[fold], N, T, B, criterion))
    
    print("Found", len(T_star), "best policies")
    
    augmented_train_loader,_ = getDataLoader(dataset_name, batch_size,transform=T_star)
    
    

        
        
        
        
    
    

    