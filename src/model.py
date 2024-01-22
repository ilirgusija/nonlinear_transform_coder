import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class Quantizer_Gaussian(nn.Module):
    def __init__(self, N_input=5, N_bottleneck=5, N_output=5):
        super(Quantizer_Gaussian, self).__init__()
        self.delta = 1.0/2.0
        N2 = 2
        # N3=100
        self.fc1 = nn.Linear(N_input, N2)
        # self.fc2 = nn.Linear(N2, N3)
        self.fc2 = nn.Linear(N2, N_bottleneck)

        self.fc3 = nn.Linear(N_bottleneck, N2)
        # self.fc5 = nn.Linear(N3, N2)
        self.fc4 = nn.Linear(N2, N_output)

    def uniform_quantizer(self, x):
        # Define the range of input values
        quantized_values = torch.round(x + 0.5)
        return quantized_values

    def transform(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        # X = F.relu(X)
        # X = self.fc2(X)
        X = F.sigmoid(X)
        return X

    def inverse_transform(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        # X = self.fc5(X)
        # X = F.relu(X)
        X = self.fc4(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        if self.training:
            noise = (torch.rand_like(features) - 0.5) * self.delta
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = self.uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized


# TODO: Implement a model that handles MNIST samples (this will be first step to eventually training on XRAYs)
class Quantizer_Images(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(Quantizer_Images, self).__init__()
        self.delta = 1.0/2.0
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)

        self.input_type = (1, 28*28)

    def uniform_quantizer(self, x):
        # Define the range of input values
        quantized_values = torch.round(x + 0.5)
        return quantized_values

    def transform(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.sigmoid(X)
        return X

    def inverse_transform(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = F.sigmoid(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        if self.training:
            noise = (torch.rand_like(features) - 0.5) * self.delta
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = self.uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized
