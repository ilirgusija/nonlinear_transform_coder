import torch
import torch.nn as nn

def device_manager(model):
    # Setup for DataParallel
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Check if multiple GPUs are available
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
        
        # Wrap models with DataParallel if more than one GPU is available
        if num_gpus > 1:
            quantizer = nn.DataParallel(quantizer)
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return model, device