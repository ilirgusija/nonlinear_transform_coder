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
from torch.distributions import Normal
from utils import device_manager
import datetime

def train(model, epochs, optimizer, scheduler, lambda_, pdf_std, data_loader, device):
    print("training...")
    model.train()  # Set the model to training mode
    model.to(device)
    losses_train = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in data_loader:
            inputs = batch[0].to(device)
            
            # Forward pass: compute the predicted outputs
            outputs, quantized = model(inputs)
            
            # Compute loss
            dist_loss = nn.MSELoss()(outputs, inputs)
            rate_loss = rate_loss_fn(quantized, pdf_std)
            loss = dist_loss + lambda_ * rate_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # Save the loss
        losses_train += [epoch_loss/len(data_loader)]    

        # Print epoch statistics
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, epoch_loss/len(data_loader)))

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return torch.tensor(x, dtype=torch.float32)

# Rate Loss Function
def rate_loss_fn(quantized_values, std=1.0):
    normal_dist = Normal(0, std)
    # Calculating negative log-likelihood as a proxy for rate
    nll = -normal_dist.log_prob(quantized_values)
    return torch.mean(nll)

def main():
    batch_size = 10
    epochs = 50
    M = 10000
    data = torch.randn(M, M)
    pdf_std = 1.0
    
    # Define the loss weight
    lambda_ = [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10]
    for idx, l_ in enumerate(lambda_):
        # Apply the transform and create a dataset
        transform = transforms.Lambda(to_tensor)
        tensor_data = transform(data)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(tensor_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # for n_levels in n_levels_list:
        quantizer = Quantizer_Gaussian(N_input=M, N_bottleneck=10, N_output=M)    # Initialize the model
        quantizer, device = device_manager(quantizer)
        
        # Define the optimizer and scheduler
        optimizer = optim.Adam(quantizer.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        losses_train = train(quantizer, epochs, optimizer, scheduler, l_, pdf_std, data_loader, device)
        
        # Plot loss curve
        plt.plot(losses_train)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss Curve')
        plot_path = f'../plots/quantizer_gauss_{idx}.png'
        plt.savefig(plot_path)
        plt.close()
        save_path = f'../params/quantizer_gauss_params_{idx}.pth'
        torch.save(quantizer.state_dict(), save_path)
        print(f"Saved trained model to {save_path}")


if __name__ == "__main__":
    # Call the main function with parsed arguments
    main()