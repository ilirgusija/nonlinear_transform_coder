
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import compressai
from compressai.models import ScaleHyperprior

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

import numpy as np

from ../utils import uniform_quantizer

class MNIST_VAE(nn.Module):
    def __init__(self):
        super(MNIST_VAE, self).__init__()
        N = 192
        M = 320

        self.relu = nn.ReLU()

        # Analysis Layers
        self.input_layer_a =  nn.Conv2d(1, N, kernel_size=5, stride=2, padding=1, bias=False),
        self.intermediate_layer_a = nn.Conv2d(N, N, kernel_size=5, stride=2, padding=1, bias=False),
        self.out_layer_a = nn.Conv2d(N, M, kernel_size=5, stride=2, padding=1, bias=False),
        
        # Synthesis Layers
        self.input_layer_s = nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=1, bias=False)
        self.intermediate_layer_s = nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, bias=False)
        self.output_layer_s = nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, padding=1, bias=False)

        # Hyperprior Analysis Layers
        self.in_layer_ha = nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_ha = nn.Conv2d(N, N, kernel_size=5, stride=2, padding=1, bias=False)
        
        # Hyperprior Synthesis Layers
        self.layer_hs = nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, bias=False)
        self.out_layer_hs = nn.Conv2d(N, M, kernel_size=3, stride=1, padding=1, bias=False)

    def analysis(self, X):
        X = self.input_layer_a(X)
        X = GDN(X)
        X = self.intermediate_layer_a(X)
        X = GDN(X)
        X = self.intermediate_layer_a(X)
        X = GDN(X)
        X = self.out_layer_a(X)
        return X

    def synthesis(self, X):
        X = self.input_layer_s(X)
        X = IGDN(X)
        X = self.intermediate_layer_s(X)
        X = IGDN(X)
        X = self.intermediate_layer_s(X)
        X = IGDN(X)
        X = self.output_layer_s(X)
        return X

    def hyperprior_synthesis(self, X):
        X = abs(X)
        X = self.in_layer_ha(X)
        X = self.relu(X)
        X = self.layer_ha(X)
        X = self.relu(X)
        X = self.layer_ha(X)
        return X

    def hyperprior_analyisis(self,X):
        X = self.layer_hs(X)
        X = self.relu(X)
        X = self.layer_hs(X)
        X = self.relu(X)
        X = self.out_layer_hs(X)
        X = self.relu(X)
        return X

    def forward(self, x):
        y = self.analysis(x)
        z = self.hyperprior_analyisis(y)
        noise_y = torch.rand_like(y) - 0.5
        noise_z = torch.rand_like(z) - 0.5
        if self.training:
            y_hat = y + noise_y
            z_hat = z + noise_z
        else:
            y_hat = uniform_quantizer(y)
            z_hat = uniform_quantizer(z)

        x_hat = self.synthesis(y_hat)


def main():
    b = 64
    train_loader = DataLoader(dataset, batch_size=b, shuffle=True)
    # img = dataset[0]["img"]
    # plt.imshow(np.squeeze(img))
    # plt.show()
    return 0

if __name__ == "__main__":
    main()
