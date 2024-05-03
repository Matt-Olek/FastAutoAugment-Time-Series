import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_WandB = False      # Use WandB for logging (True/False)