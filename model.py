import torch
from torch import nn
import numpy as np

class Quantizer(nn.Module):
    
    def __init__(self, data, n_levels):
        super(Quantizer, self).__init__()
        self.levels = nn.Parameter(torch.linspace(data.min(), data.max(), n_levels))
        
    def encode(self, X):
        # Calculate distances to quantization levels
        distances = torch.abs(X.unsqueeze(1) - self.levels)
        # print('Distances {}'.format(distances))
        # Get the index of the nearest level
        indices = torch.argmin(distances, dim=-1)
        # print('Indices {}'.format(indices))
        return indices

    def decode(self, indices):
        # Map indices back to quantization levels
        # print('Levels[Indices] {}'.format(self.levels[indices]))
        return self.levels[indices] 

    def forward(self, X):
        # Encode then decode the input
        indices = self.encode(X)
        quantized = self.decode(indices)
        return quantized