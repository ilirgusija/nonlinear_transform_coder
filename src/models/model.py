import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import zlib
import bz2
import lzma
import pickle
from utils import uniform_quantizer

class Quantizer_Gaussian(nn.Module):
    def __init__(self, N_input=1, N_bottleneck=7, N_output=1):
        super(Quantizer_Gaussian, self).__init__()
        self.delta = 1.0/2.0
        N2 = 3
        N3 = 5
        # N2 = int(N_input/2)
        # N3= int(N2/2)
        self.fc1_e = nn.Linear(N_input, N2)
        self.fc2_e = nn.Linear(N2, N3)
        self.fc3_e = nn.Linear(N3, N_bottleneck)

        self.fc1_d = nn.Linear(N_bottleneck, N3)
        self.fc2_d = nn.Linear(N3, N2)
        self.fc3_d = nn.Linear(N2, N_output)

    def transform(self, X):
        X = self.fc1_e(X)
        X = F.relu(X)
        X = self.fc2_e(X)
        X = F.relu(X)
        X = self.fc3_e(X)
        X = F.relu(X)
        return X

    def inverse_transform(self, X):
        X = self.fc1_d(X)
        X = F.relu(X)
        X = self.fc2_d(X)
        X = F.relu(X)
        X = self.fc3_d(X)
        return X

    def forward(self, X):
        # Encode then decode the input
        features = self.transform(X)
        if self.training:
            noise = torch.rand_like(features) - 0.5
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized

class MNIST_Coder(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784, compression_method='zlib'):
        super(MNIST_Coder, self).__init__()
        N2 = 392
        N3 = int(N2/2)
        
        self.fc1_e = nn.Linear(N_input, N2)
        self.fc2_e = nn.Linear(N2, N_bottleneck)
        # self.fc3_e = nn.Linear(N3, N_bottleneck)
        
        # self.fc1_d = nn.Linear(N_bottleneck, N3)
        self.fc2_d = nn.Linear(N_bottleneck, N2)
        self.fc3_d = nn.Linear(N2, N_output)
        
        self.compression_method = compression_method

    def transform(self, X):
        X = self.fc1_e(X)
        X = F.relu(X)
        X = self.fc2_e(X)
        # X = F.relu(X)
        # X = self.fc3_e(X)
        X = F.relu(X)
        return X

    def inverse_transform(self, X):
        # X = self.fc1_d(X)
        # X = F.relu(X)
        X = self.fc2_d(X)
        X = F.relu(X)
        X = self.fc3_d(X)
        X = F.sigmoid(X)
        return X
    
    def encode(self, data):
        # Serialize the quantized data from Tensor to a format that our encoder can handle
        data = pickle.dumps(data)
        
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
            data = zlib.decompress(data)
        elif self.compression_method == 'bz2':
            data = bz2.decompress(data)
        elif self.compression_method == 'lzma':
            data = lzma.decompress(data)
        else:
            raise ValueError("Unsupported decompression method")
        
        # Deserialize the output of our quantizer
        data = pickle.loads(data)
        return data
    
    def compress(self, X):
        # Run the neural network
        features = self.transform(X)
        
        # Quantize our features
        quantized = uniform_quantizer(features)
        
        # Losslessly code using the method defined at initialization
        encoded = self.encode(quantized)
        return encoded

    def decompress(self, X):
        # Run the decoder on our compressed file
        decoded = self.decode(X)
        
        # Run neural network
        output = self.inverse_transform(decoded)
        
        return output

    def forward(self, X):
        features = self.transform(X)
        if self.training:
            noise = torch.rand_like(features) - 0.5
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized

class MNIST_FCNN(nn.Module):
    def __init__(self):
        super(MNIST_FCNN, self).__init__()
        # Encoder layers
        self.transform = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # Output: (32, 7, 7)
            nn.ReLU(), 
            nn.Conv2d(32, 64, 7)                      # Output: (64, 1, 1)
        )
        # Decoder layers
        self.inverse_transform = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),            # Output: (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # Output: (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 28)
            nn.Sigmoid()  # Using sigmoid for output to get values between 0 and 1
        )

    def forward(self, X):
        features = self.transform(X)
        if self.training:
            noise = torch.rand_like(features) - 0.5
            quantized = features + noise
            output = self.inverse_transform(quantized)
        else:
            quantized = uniform_quantizer(features)
            output = self.inverse_transform(quantized)
        return output, quantized

class MNIST_VAE(nn.Module):
    def __init__(self):
        super(MNIST_VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
        )

    def analysis(self, X):
        return 0

    def synthesis(self, X):
        return 0

    


    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
