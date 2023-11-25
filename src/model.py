import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class Quantizer_Gaussian(nn.Module):
    
    def __init__(self, n_levels, N_input=10000, N_bottleneck=10, N_output=10000):
        super(Quantizer_Gaussian, self).__init__()
        self.n_levels=n_levels
        self.delta=1.0/float(n_levels)
        N2=1000
        N3=100
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N3)
        self.fc3 = nn.Linear(N3, N_bottleneck)
        
        self.fc4 = nn.Linear(N_bottleneck,N3)
        self.fc5 = nn.Linear(N3, N2)
        self.fc6 = nn.Linear(N2, N_output)
        
    def uniform_quantizer(self, x):
        # Define the range of input values
        quantized_values = torch.round((x) / self.delta) * self.delta
        return quantized_values
    
    def transform(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.sigmoid(X)
        return X

    def inverse_transform(self, X):
        X = self.fc4(X)
        X = F.relu(X)
        X = self.fc5(X)
        X = F.relu(X)
        X = self.fc6(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        if self.training:
            noise = (torch.rand_like(features) - 0.5) * self.delta
            features = features + noise 
            quantized = self.inverse_transform(features)
        else:
            quantized = self.uniform_quantizer(features)
        return quantized
    
    
# TODO: Implement a model that handles MNIST samples (this will be first step to eventually training on XRAYs)
class Quantizer_Images(nn.Module):
    
    def __init__(self, data, n_levels, N_input=1, N_bottleneck=10, N_output=1):
        super(Quantizer_Gaussian, self).__init__()
        self.n_levels=n_levels
        self.delta=calculate_delta(data, n_levels)
        # N2=100
        N2=50
        N3=25
        self.fc1 = nn.Linear(N_input, N2)
        # self.fc2 = nn.Linear(N2, N2)
        self.fc3 = nn.Linear(N2, N3)
        self.fc4 = nn.Linear(N3, N_bottleneck)
        
        self.fc5 = nn.Linear(N_bottleneck,N3)
        self.fc6 = nn.Linear(N3, N2)
        # self.fc7 = nn.Linear(N2, N2)
        self.fc8 = nn.Linear(N2, N_output)
        # self.input_type = (1, 28*28)
        
    def transform(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        # X = self.fc2(X)
        # X = F.relu(X)
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = F.sigmoid(X)
        return X

    def inverse_transform(self, X):
        X = self.fc5(X)
        X = F.relu(X)
        X = self.fc6(X)
        X = F.relu(X)
        # X = self.fc7(X)
        # X = F.relu(X)
        X = self.fc8(X)
        # X = F.relu(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        noise = (torch.rand_like(features) - 0.5) * self.delta
        noisy_features = features + noise 
        quantized = self.inverse_transform(noisy_features)
        return quantized