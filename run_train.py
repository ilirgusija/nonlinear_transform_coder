import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import Quantizer
from train import train
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def main(l):
    n_levels_list=[1, 2, 4, 8, 16]
    batch_size = 100
    epochs = 100
    M = 10000
    data = torch.randn(M, 1)
    loss_fn = nn.MSELoss()
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for n_levels in n_levels_list:
        quantizer = Quantizer(data, n_levels)
        
        optimizer = optim.SGD(quantizer.parameters(), lr=0.01)
        
        loss = train(quantizer, epochs, optimizer, loss_fn, data_loader, batch_size, 'cpu')
        
        save_path = f'quantizer_params_{n_levels}.pt'
        torch.save(quantizer.state_dict(), save_path)
        print(f"Saved trained model with {n_levels} levels to {save_path}")
    

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Training script")

    # Add arguments
    parser.add_argument("-l", type=str, required=True, help="The l argument")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.l)