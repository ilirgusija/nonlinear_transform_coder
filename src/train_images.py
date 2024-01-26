import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import MNIST_Coder
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import datetime
from utils import device_manager, calc_distortion, calc_normal_rate, compute_loss

def train(model, epochs, optimizer, scheduler, loss_fn, data_loader, device, early_stopping_rounds=10):
    print('training ...')
    model.train()  # Set the model to training mode
    model.to(device)
    losses_train = []
    
    # Early stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in data_loader:
            imgs = imgs.view(imgs.size(0), -1) # Flatten the imgs
            inputs = imgs.to(device)
            
            # Forward pass: compute the predicted outputs
            outputs, quantized = model(inputs)
            
            # Compute loss
            loss = compute_loss(loss_fn, inputs, outputs, quantized)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
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
    batch_size = 128
    epochs = 50
    
    # Define the loss weights
    lambda_ = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
    # lambda_ = [0.001]
    
    data_loader = DataLoader(datasets.MNIST('../data/mnist',
                                             train=True,
                                             download=True,
                                             transform=transforms.Compose([transforms.ToTensor()])),
                              batch_size=batch_size,
                              shuffle=True)
    
    for idx, l_ in enumerate(lambda_):
        model = MNIST_Coder()    # Initialize the model

        model, device = device_manager(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        loss_fn = lambda img, out, q: (calc_distortion(img, out) + l_*calc_normal_rate(q))
        # loss_fn = nn.MSELoss()
        
        losses_train = train(model, epochs, optimizer, scheduler, loss_fn, data_loader, device)
        
        # Plot loss curve
        plt.plot(losses_train)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss Curve')
        plot_path = f'../plots/model_MNIST_{l_}.png'
        plt.savefig(plot_path)
        plt.close()
        
        save_path = f'../params/model_MNIST_params_{l_}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Saved trained model to {save_path}")
    
if __name__ == "__main__":
    # Call the main function with parsed arguments
    main()