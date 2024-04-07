import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import inspect
import pynvml

# Function to get the memory info
def get_free_memory():
    device_count = pynvml.nvmlDeviceGetCount()
    free_memory = []
    for device_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory.append(meminfo.free)
    return free_memory

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def device_manager(model=None):
    # Setup for DataParallel
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Check if multiple GPUs are available
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
            
        if model is not None:
            # Wrap models with DataParallel if more than one GPU is available
            if num_gpus > 1:
                print("Moving device to two GPUs")
                model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    if model is None:
        return device
    else:
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
def calc_normal_rate(quantized_values, stds):
    # Ensure stds is not zero to avoid division by zero
    stds = stds.clamp(min=1e-6)
    normal_dist = torch.distributions.Normal(0, stds)
    nll = -normal_dist.log_prob(quantized_values)
    return torch.mean(nll)

def calc_distortion(orig, output):
    dist = nn.MSELoss()
    return dist(orig,output)

def init_state(num_elements):
    """Initialize the computation state."""
    return (torch.zeros(num_elements),  # sum_x
            torch.zeros(num_elements),  # sum_x_squared
            0)  # n

def update_state(state, batch):
    """Update the state with a new batch of data."""
    sum_x, sum_x_squared, n = state
    new_n = n + batch.shape[0]
    new_sum_x = sum_x + torch.sum(batch, dim=0)
    new_sum_x_squared = sum_x_squared + torch.sum(batch ** 2, dim=0)
    return (new_sum_x, new_sum_x_squared, new_n)

def calculate_variance(state):
    """Calculate the variance for each element based on the current state."""
    sum_x, sum_x_squared, n = state
    mean_square = sum_x_squared / n
    square_mean = (sum_x / n) ** 2
    variance = (n * mean_square - sum_x ** 2) / (n * (n - 1))
    return variance

def gen_gaussian_data(x, y, batch_size):
    data = torch.randn(x, y)
    # Apply the transform and create a dataset
    transform = transforms.Lambda(to_tensor)
    tensor_data = transform(data)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
