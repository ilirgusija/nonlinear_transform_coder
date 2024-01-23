import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import zlib
import bz2
import lzma
import pickle

class Quantizer_Gaussian(nn.Module):
    def __init__(self, N_input=5, N_bottleneck=5, N_output=5):
        super(Quantizer_Gaussian, self).__init__()
        self.delta = 1.0/2.0
        N2 = int(N_input/2)
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
        X = F.relu(X)
        return X

    def inverse_transform(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        if self.training:
            noise = torch.rand_like(features)
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = self.uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized

class MNIST_Coder(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=10, N_output=784, compression_method='zlib'):
        super(MNIST_Coder, self).__init__()
        self.delta = 1.0/2.0
        N2 = 392
        N3 = int(N2/2)
        self.fce1 = nn.Linear(N_input, N2)
        self.fce2 = nn.Linear(N2, N3)
        self.fce3 = nn.Linear(N3, N_bottleneck)
        self.fcd1 = nn.Linear(N_bottleneck, N3)
        self.fcd2 = nn.Linear(N3, N2)
        self.fcd3 = nn.Linear(N2, N_output)
        self.compression_method = compression_method

        self.input_type = (1, 28*28)

    def uniform_quantizer(self, x):
        # Define the range of input values
        quantized_values = torch.round(x + 0.5)
        return quantized_values

    def transform(self, X):
        X = self.fce1(X)
        X = F.relu(X)
        X = self.fce2(X)
        X = F.relu(X)
        X = self.fce3(X)
        X = F.relu(X)
        return X

    def inverse_transform(self, X):
        X = self.fcd1(X)
        X = F.relu(X)
        X = self.fcd2(X)
        X = F.relu(X)
        X = self.fcd3(X)
        X = F.sigmoid(X)
        return X
    
    def encode(self, data):
        if self.compression_method == 'zlib':
            return zlib.compress(data)
        elif self.compression_method == 'bz2':
            return bz2.compress(data)
        elif self.compression_method == 'lzma':
            return lzma.compress(data)
        else:
            raise ValueError("Unsupported compression method")

    def decode(self, data):
        if self.compression_method == 'zlib':
            return zlib.decompress(data)
        elif self.compression_method == 'bz2':
            return bz2.decompress(data)
        elif self.compression_method == 'lzma':
            return lzma.decompress(data)
        else:
            raise ValueError("Unsupported decompression method")
    
    def compress(self, X):
        features = self.transform(X)
        quantized = self.uniform_quantizer(features)
        serialized_quantized = pickle.dumps(quantized)
        encoded = self.encode(serialized_quantized)
        return encoded

    def decompress(self, X):
        decoded = self.decode(X)
        serialized_output = pickle.loads(decoded)
        output = self.inverse_transform(serialized_output)
        return output

    def forward(self, X):
        # Encode then decode the input
        if self.training:
            features = self.transform(X)
            noise = (torch.rand_like(features) - 0.5) * self.delta
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = self.compress(X)
            output = self.decompress(quantized)
        return output, quantized
