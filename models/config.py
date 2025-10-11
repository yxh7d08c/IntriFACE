import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 85742
