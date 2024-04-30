import torch 
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ ResNet ------------------------------ #

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)                  # Convolution 1
        self.bn1 = nn.BatchNorm1d(out_channels)                                                                 # Batch Normalization 1
        self.relu = nn.ReLU()                                                                                   # ReLU 1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)                 # Convolution 2  
        self.bn2 = nn.BatchNorm1d(out_channels)                                                                 # Batch Normalization 2

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(                                                                      # Shortcut connection
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.initialize_weights()

    def forward(self, x):
        residual = x
        out = self.conv1(x)                     # Convolution 1
        out = self.bn1(out)                     # Batch Normalization 1
        out = self.relu(out)                    # ReLU 1
        out = self.conv2(out)                   # Convolution 2
        out = self.bn2(out)                     # Batch Normalization 2
        out += self.shortcut(residual)          # Residual connection
        out = self.relu(out)                    # ReLU 2
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Classifier_RESNET(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(Classifier_RESNET, self).__init__()

        n_feature_maps = 64
        self.conv1 = nn.Conv1d(1, n_feature_maps, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(n_feature_maps)
        self.relu = nn.ReLU()

        self.residual_block1 = ResidualBlock(n_feature_maps, n_feature_maps, kernel_size=3)                 # Residual Block 1
        self.residual_block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_size=3)             # Residual Block 2
        self.residual_block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3)         # Residual Block 3

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)
        
        self.initialize_weights()

    def forward(self, x):
        out = self.conv1(x)                                     # Convolution 1
        out = self.bn1(out)                                     # Batch Normalization 1
        out = self.relu(out)                                    # ReLU 1
        out = self.residual_block1(out)                         # Residual Block 1
        out = self.residual_block2(out)                         # Residual Block 2
        out = self.residual_block3(out)                         # Residual Block 3
        out = self.global_avg_pool(out)                         # Global Average Pooling
        out = torch.flatten(out, 1)                             # Flatten
        out = self.fc(out)                                      # Fully Connected
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
# ------------------------------ Utils ------------------------------ #

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
