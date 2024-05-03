import numpy as np
from tsaug import Crop, Drift, Reverse, AddNoise, Resize  

possible_transformations = {
    'Crop' : lambda series, magnitude: crop(series, magnitude),
    'Drift' : lambda series, magnitude: drift(series, magnitude),
    'AddNoise' : lambda series, magnitude: add_noise(series, magnitude),
    'Reverse' : lambda series, magnitude: reverse(series, magnitude)
}



class Policy:
    def __init__(self):
        self.transformation_name = possible_transformations[np.random.randint(0, len(possible_transformations))]
        self.probability = np.random.uniform(0, 1)
        self.magnitude = np.random.uniform(0, 1)
        
        
    def __str__(self):
        return f"Transformation: {self.transformation_name}, Probability: {self.probability}, Magnitude: {self.magnitude}"
    
    
def sample_policy(p, lambda_):
    policy = Policy()
    policy.probability = p
    policy.magnitude = lambda_
    return policy

def apply_policy_to_dataset(dataset, policies):
    num_policies = len(policies)/3
    
    output = []
    for data in dataset:
        for policy in policies:
            X, y = apply_policy(data, policy)
            output.append((X, y))
    return output

def apply_policy(data, policy):
    X, y = data
    print(policy)
    prob = policy.probability
    magnitude = policy.magnitude
    if np.random.uniform(0, 1) < prob:
        output = possible_transformations[policy.transformation_name](X, magnitude)
    else :
        output = X
    return output, y
        

# ------------------- Transformations ------------------- #

def crop(series_XY, magnitude):
    max_size = len(series_XY[0])
    min_size = 0
    magnitude = round(min_size + magnitude * (max_size - min_size))
    X = series_XY[0]
    Y = series_XY[1]
    croped = Crop(size=magnitude).augment(X, Y)
    return Resize(size=len(X)).augment(croped[0], croped[1])

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