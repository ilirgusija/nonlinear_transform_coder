import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import Quantizer_Gaussian
from train_images import train
import torch.optim as optim
import torch
from utils import device_manager, calc_distortion, calc_rate, gen_gaussian_data
import datetime

def train(model, epochs, optimizer, scheduler, lambda_, pdf_std, data_loader, device, early_stopping_rounds=10):
    print("training...")
    model.train()
    model.to(device)
    losses_train = []
    
    # Early stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            inputs = batch[0].to(device) 
            
            # Forward pass: compute the predicted outputs
            outputs, quantized = model(inputs)
            
            # Compute loss
            dist_loss = calc_distortion(outputs, inputs)
            rate_loss = calc_rate(quantized, pdf_std)
            loss = dist_loss + lambda_ * rate_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / len(data_loader)
        losses_train.append(avg_epoch_loss)
        
        # Print epoch statistics
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, avg_epoch_loss))
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_rounds:
            print('Early stopping triggered after epoch {}'.format(epoch))
            break
    return losses_train

def main():
    batch_size = 10
    epochs = 100
    M = 10000
    pdf_std = 1.0
    
    data_loader = gen_gaussian_data(M, 1, batch_size)
    
    # Define the loss weight
    lambda_ = [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10]
    for idx, l_ in enumerate(lambda_):
        quantizer = Quantizer_Gaussian(N_input=1, N_bottleneck=10, N_output=1)    # Initialize the model
        quantizer, device = device_manager(quantizer) # Move model to necessary device and return the device type
        
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