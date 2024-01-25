import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

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

def uniform_quantizer(x):
    # Define the range of input values
    quantized_values = torch.round(x)
    return quantized_values

def calc_rate(quantized_values, std=1.0):
    normal_dist = Normal(0, std)
    # Calculating negative log-likelihood as a proxy for rate
    nll = -normal_dist.log_prob(quantized_values)
    return torch.mean(nll)

def calc_distortion(orig, output):
    dist = nn.MSELoss()
    return dist(orig,output)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return torch.tensor(x, dtype=torch.float32)

def gen_gaussian_data(x, y, batch_size):
    data = torch.randn(x, y)
    # Apply the transform and create a dataset
    transform = transforms.Lambda(to_tensor)
    tensor_data = transform(data)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader