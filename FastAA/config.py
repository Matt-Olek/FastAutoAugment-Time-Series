import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_WandB = True      # Use WandB for logging