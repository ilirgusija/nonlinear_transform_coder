import datetime
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from model import MLP
from torchvision import transforms, datasets
from torchsummary import summary
from torch.utils.data import DataLoader


# -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
# -z=bottleneck -e=epochs -b=batch_size -p=path

def train(model, data_loader, optimizer, epochs, loss_fn, device):
    model.train()  # Set the model to training mode
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        for inputs, _ in data_loader:

            # Forward pass: compute the predicted outputs
            quantized_outputs = model(inputs)

            # Compute loss
            loss = loss_fn(quantized_outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        # Print epoch statistics
        epoch_loss /= len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return loss