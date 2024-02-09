import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import inspect

def compute_loss(loss_fn, img, out, q):
    # Get the argument count of the loss function
    if callable(loss_fn):
        loss_fn_args = inspect.signature(loss_fn).parameters
        if len(loss_fn_args) == 3:
            # Custom lambda loss function
            return loss_fn(img, out, q)
        elif len(loss_fn_args) == 2:
            # Standard loss function like nn.MSELoss
            return loss_fn(img, out)
        else:
            raise ValueError("Unsupported loss function signature")
    else:
        raise TypeError("loss_fn is not a callable function")

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
            model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return model.to(device), device

def uniform_quantizer(x):
    # Define the range of input values
    quantized_values = torch.round(x)
    return quantized_values

def calc_empirical_pmf(X):
    # Flatten the tensor to a 1D array for simplicity
    flattened = X.flatten()

    # Get unique values and their counts
    unique_values, counts = torch.unique(flattened, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts.float() / flattened.numel()
    
    return probabilities

def calc_empirical_rate(X):
    probabilities = calc_empirical_pmf(X)
    # Calculate the entropy
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy

# TODO Change std based on empirical rate of input data
def calc_normal_rate(quantized_values, std=1.0):
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
