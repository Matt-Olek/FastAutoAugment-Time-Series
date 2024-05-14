import torch
from config import device
from collections import OrderedDict
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

# ------------------------------ Validation ------------------------------ #

def validate(test_dataset, model, criterion):
    
    model.eval()
    X_test, y_test = test_dataset
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2]).float()
    
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred.reshape(y_test.shape).float()
        loss = criterion(y_pred, y_test)
        acc = accuracy_score(torch.argmax(y_test, dim=1).cpu().numpy(), torch.argmax(y_pred, dim=1).cpu().numpy())
        f1 = f1_score(torch.argmax(y_test, dim=1).cpu().numpy(), torch.argmax(y_pred, dim=1).cpu().numpy(), average='macro')
        precision = precision_score(torch.argmax(y_test, dim=1).cpu().numpy(), torch.argmax(y_pred, dim=1).cpu().numpy(), average='macro', zero_division=0)
        recall = recall_score(torch.argmax(y_test, dim=1).cpu().numpy(), torch.argmax(y_pred, dim=1).cpu().numpy(), zero_division=0, average='macro')
        
    log = OrderedDict({'loss': loss.item(), 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall})
    return log
