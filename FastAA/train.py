# Defines the training process for the baseline and augmented models

from loader import getDataLoader
from model import Classifier_RESNET
from config import device, is_WandB
from validation import validate
from torch import nn, optim
from torch.optim import lr_scheduler
import pandas as pd
import wandb
from tqdm import tqdm
import torch

def train_baseline(dataset_name, nb_classes,epochs,batch_size,lr=0.001, weight_decay=0.0001):
    '''
    Defines the training process for the baseline and augmented models
    '''
    # WandB - Initialize a new run
    if is_WandB:
        wandb.init(
            project='FastAA', 
            config={'dataset': dataset_name, 
                    'nb_classes': nb_classes, 
                    'epochs': epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay}
            )
    
    # Load the dataset
    train_loader, test_loader = getDataLoader(dataset_name, batch_size)
    
    # Define the model
    model = Classifier_RESNET(input_shape=train_loader.dataset[0][0].shape, nb_classes=nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    
    # Training
    best_acc = 0
    for epoch in tqdm(range(epochs)):
        epoch_avg_loss, epoch_accuracy = train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch)
        val_log = validate(test_loader, model, criterion)

        # WandB - Log metrics
        if is_WandB:
            wandb.log({'epoch': epoch, 'train_loss': epoch_avg_loss, 'train_accuracy': epoch_accuracy, 'val_loss': val_log['loss'], 'val_accuracy': val_log['acc']})
        
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
            
            
    

    