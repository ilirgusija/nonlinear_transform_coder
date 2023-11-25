import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import Quantizer_Gaussian
from train_images import train
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import sys

def train(model, epochs, optimizer, scheduler, loss_fn, data_loader, device):
    model.train()  # Set the model to training mode
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in data_loader:
            inputs = batch[0].to(device)
            
            # Forward pass: compute the predicted outputs
            quantized_outputs = model(inputs)
            
            # Compute loss
            loss = loss_fn(quantized_outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # Print epoch statistics
        epoch_loss /= len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return torch.tensor(x, dtype=torch.float32)

def main():
    n_levels_list=[1, 2, 4, 8, 16]
    # n_levels_list=[4]
    batch_size = 10
    epochs = 100
    M = 10000
    data = torch.randn(M, M)
    loss_fn = nn.MSELoss()
    
    
    # Apply the transform and create a dataset
    transform = transforms.Lambda(to_tensor)
    tensor_data = transform(data)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for n_levels in n_levels_list:
        quantizer = Quantizer_Gaussian(n_levels, N_input=M, N_output=M)
        
        optimizer = optim.Adam(quantizer.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        train(quantizer, epochs, optimizer, scheduler, loss_fn, data_loader, 'cuda')
        
        save_path = f'../params/quantizer_params_{n_levels}.pt'
        torch.save(quantizer.state_dict(), save_path)
        print(f"Saved trained model with {n_levels} levels to {save_path}")
    

if __name__ == "__main__":

    # Call the main function with parsed arguments
    main()