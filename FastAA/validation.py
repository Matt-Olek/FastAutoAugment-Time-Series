import utils 
import torch
from config import device
from collections import OrderedDict

# ------------------------------ Validation ------------------------------ #

def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device).float()
            target = target.to(device).float()
            output = model(input).float()
            loss = criterion(output, target)
            acc = torch.sum(torch.argmax(output, dim=1) == torch.argmax(target, dim=1))
            losses.update(loss.item(), input.size(0))
            scores.update(acc.item(), input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])
    return log
