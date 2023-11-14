import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import Quantizer
from train import train
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    data = torch.randn(M, 1)
    loss_fn = nn.MSELoss()
    
    
    # Apply the transform and create a dataset
    transform = transforms.Lambda(to_tensor)
    tensor_data = transform(data)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for n_levels in n_levels_list:
        quantizer = Quantizer(data, n_levels)
        
        optimizer = optim.Adam(quantizer.parameters(), lr=0.01, weight_decay=1e-5)
        
        train(quantizer, epochs, optimizer, loss_fn, data_loader, 'cpu')
        
        save_path = f'quantizer_params_{n_levels}.pt'
        torch.save(quantizer.state_dict(), save_path)
        print(f"Saved trained model with {n_levels} levels to {save_path}")
    

if __name__ == "__main__":

    # Call the main function with parsed arguments
    main()