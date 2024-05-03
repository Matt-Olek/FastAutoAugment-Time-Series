from hyperopt import fmin, tpe, hp, Trials
from transformations import possible_transformations, apply_policy_to_dataset
import numpy as np
from torch import nn

num_opt = 2
K=5
K_ratio = 0.2   
N=10

def bayesian_optimization(model,train_fold, N, T, B, criterion,sub_policies=1):
    model.eval()
    space = get_policy_space(B, num_opt)
    final_policy_set = []
    for _ in range(T):
        for fold in range(K):
            name = "search_fold_%d" % fold            
            print(name)
            trials = Trials()
            best = fmin(lambda policies: eval_tta(model,train_fold, policies),
                        space=space, algo=tpe.suggest, max_evals=4, trials=trials)
            results = sorted(trials.results, key=lambda x: x['loss'])
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
    return space

def eval_tta(model, train_fold, augs):
    criterion = nn.CrossEntropyLoss()
    augmented_fold = apply_policy_to_dataset(train_fold, augs)
    loss_on_augmented = criterion(model(augmented_fold[0]), augmented_fold[1])
    return {'loss': loss_on_augmented.item(), 'policy': augs}