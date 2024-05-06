from hyperopt import fmin, tpe, hp, Trials
from transformations import possible_transformations, apply_policy_to_dataset
import numpy as np
from torch import nn
import torch

num_opt = 2
K=5
K_ratio = 0.2   
N=10

def bayesian_optimization(model,train_fold, N, T, B, criterion,sub_policies=1):
    model.eval()
    space, space_shape = get_policy_space(B, num_opt)
    final_policy_set = []
    for _ in range(T):
        for fold in range(K):
            name = "search_fold_%d" % fold            
            trials = Trials()
            best = fmin(lambda policies: eval_tta(model,train_fold, policies, space_shape),
                        space=space, algo=tpe.suggest, max_evals=4, trials=trials)
            results = sorted(trials.results, key=lambda x: x['loss'])
            print(results)
            for result in results[:N]:
                final_policy_set.append(result['policy'])
    return final_policy_set
                
def get_policy_space(num_policy, num_op):
    space = {}
    ops = possible_transformations
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)
    space_shape = (num_policy, num_op)
    return space, space_shape

def eval_tta(model, train_fold, augs, space_shape):
    criterion = nn.CrossEntropyLoss()
    augmented_fold = apply_policy_to_dataset(train_fold, augs, space_shape)
    X = torch.stack([x[0] for x in augmented_fold])
    y = torch.stack([x[1] for x in augmented_fold]).float()
    loss_on_augmented = criterion(model(X), y)
    return {'loss': loss_on_augmented.item(), 'policy': augs, 'status': 'ok'}