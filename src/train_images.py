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