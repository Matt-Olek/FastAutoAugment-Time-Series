import numpy as np
import torch
from tsaug import Crop, Drift, Reverse, AddNoise, Resize  

possible_transformations = {
    'Crop' : lambda series, magnitude: crop(series, magnitude),
    'Drift' : lambda series, magnitude: drift(series, magnitude),
    'AddNoise' : lambda series, magnitude: add_noise(series, magnitude),
    'Reverse' : lambda series, magnitude: reverse(series, magnitude),
    'Identity' : lambda series, magnitude: identity(series, magnitude)
}



class Policy:
    def __init__(self):
        self.transformation_name = np.random.choice(list(possible_transformations.keys()))
        self.probability = np.random.uniform(0, 1)
        self.magnitude = np.random.uniform(0, 1)
        
        
    def __str__(self):
        return f"Transformation: {self.transformation_name}, Probability: {self.probability}, Magnitude: {self.magnitude}"
    
    
def sample_policy(p, lambda_):
    policy = Policy()
    policy.probability = p
    policy.magnitude = lambda_
    return policy

def apply_policy_to_dataset(dataset, policy, policy_idx, space_shape):  
    output = []
    i=policy_idx
    for data in dataset:
        X, y = data
        for j in range(space_shape[1]):
            policy_id = policy['policy_%d_%d' % (i, j)]
            prob = policy['prob_%d_%d' % (i, j)]
            level = policy['level_%d_%d' % (i, j)]
            policy_object = sample_policy(prob, level)
            policy_object.transformation_name = list(possible_transformations.keys())[policy_id]
            X, y = apply_policy((X, y), policy_object)
    output.append((X.clone().detach(), y))
    return output

def apply_policy(data, policy):
    X, y = data
    prob = policy.probability
    magnitude = policy.magnitude
    serie_XY = [np.linspace(1, X.size(1), X.size(1)), X.squeeze(0).numpy()]
    if np.random.uniform(0, 1) < prob:
        output = possible_transformations[policy.transformation_name](serie_XY, magnitude)
        output = torch.tensor(output[1], dtype=torch.float32).squeeze(0)
    else :
        output = X
        output = output.squeeze(0)
    output = output.unsqueeze(0)
    return output, y

def apply_multiple_policies(X_train, policies, num_opt):
    final_X_train = []
    num_policies = len(policies)//(3*num_opt)
    for data in X_train:
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        for i in range(num_policies):
            for j in range(num_opt):
                policy_name = 'policy_%d_%d' % (i, j)
                policy_prob = 'prob_%d_%d' % (i, j)
                policy_level = 'level_%d_%d' % (i, j)
                policy = Policy()
                policy.transformation_name = list(possible_transformations.keys())[policies[policy_name]]
                policy.probability = policies[policy_prob]
                policy.magnitude = policies[policy_level]
                data,y = apply_policy((data, None), policy)
        final_X_train.append(data)
    return torch.stack(final_X_train).squeeze(1).numpy()
    
        

# ------------------- Transformations ------------------- #

def crop(series_XY, magnitude):
    max_size = len(series_XY[0])
    min_size = round(max_size * 0.1)    
    magnitude = round(min_size + magnitude * (max_size - min_size))
    X = series_XY[0]
    Y = series_XY[1]
    croped = Crop(size=magnitude).augment(X, Y)
    return Resize(size=len(X)).augment(np.linspace(1, len(croped[0]), len(croped[0])), croped[1])

def drift(series_XY, magnitude):
    max_max_drift = 1
    min_max_drift = 0
    magnitude = min_max_drift + magnitude * (max_max_drift - min_max_drift)
    X = series_XY[0]
    Y = series_XY[1]
    return Drift(max_drift=magnitude).augment(X, Y)

def add_noise(series_XY, magnitude):
    max_scale = 10
    min_scale = 0 
    magnitude = min_scale + magnitude * (max_scale - min_scale)
    X = series_XY[0]
    Y = series_XY[1]
    return AddNoise(scale=magnitude).augment(X, Y)

def reverse(series_XY, magnitude):
    X = series_XY[0]
    Y = series_XY[1]
    return Reverse().augment(X, Y)

def identity(series_XY, magnitude):
    return series_XY