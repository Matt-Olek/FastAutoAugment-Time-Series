from hyperopt import fmin, tpe, hp, Trials
from transformations import possible_transformations, apply_policy_to_dataset
import numpy as np
import torch
from torch import nn
from config import device

def bayesian_optimization(model, train_fold, N, T, B, criterion, num_opt=2):
    print('Bayesian optimization on fold')
    space, space_shape = get_policy_space(B, num_opt)
    model.eval()
    intermediate_policy_set = []

    for _ in range(1):
        trials = Trials()
        best = fmin(lambda policies: eval_tta(model, train_fold, policies, space_shape),
                     space=space, algo=tpe.suggest, max_evals=5, trials=trials)
        results = sorted(trials.results, key=lambda x: x['loss'])
        best_trial = results[0]
        intermediate_policy_set.append(best_trial)

    policies_low_losses = {}
    for element in intermediate_policy_set:
        losses = element['losses']
        N_lowest_losses_indices = np.argsort(losses)[:N]

        for i in range(N):
            k = N_lowest_losses_indices[i]
            for j in range(num_opt):
                policy_name = 'policy_%d_%d' % (k, j)
                policy_prob = 'prob_%d_%d' % (k, j)
                policy_level = 'level_%d_%d' % (k, j)
                new_policy_name = 'policy_%d_%d' % (i, j)
                new_prob_name = 'prob_%d_%d' % (i, j)
                new_level_name = 'level_%d_%d' % (i, j)
                name_value = element['policy'][policy_name]
                prob_value = element['policy'][policy_prob]
                level_value = element['policy'][policy_level]
                policies_low_losses[new_policy_name] = name_value
                policies_low_losses[new_prob_name] = prob_value
                policies_low_losses[new_level_name] = level_value

    print('Bayesian optimization completed')
    print('Found', len(policies_low_losses) // (3 * num_opt), 'best policies')
    return policies_low_losses

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
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for i in range(space_shape[0]):
        sub_augs_keys =  ['policy_%d_%d' % (i, j) for j in range(space_shape[1])]
        sub_augs_keys += ['prob_%d_%d' % (i, j) for j in range(space_shape[1])]
        sub_augs_keys += ['level_%d_%d' % (i, j) for j in range(space_shape[1])]

        augmented_fold = apply_policy_to_dataset(train_fold, {key: augs[key] for key in sub_augs_keys}, policy_idx=i, space_shape=space_shape)
        augmented_fold += train_fold
        X = torch.stack([x[0] for x in augmented_fold]).to(device)
        y = torch.stack([x[1] for x in augmented_fold]).float().to(device)

        loss_on_augmented = criterion(model(X), y)
        losses.append(loss_on_augmented.item())

    return {'loss': np.sum(losses), 'policy': augs, 'status': 'ok', 'losses': losses}
