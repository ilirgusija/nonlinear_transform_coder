import argparse
import torch
import matplotlib.pyplot as plt
from model import MNIST_Coder
from torchvision import transforms, datasets
from torchvision.io import encode_jpeg, decode_jpeg
from utils import device_manager, calc_distortion, calc_rate
from torch.utils.data import DataLoader

# Generate the Rate Distortion graph for JPEG based on quality
def gen_RD_graph_for_jpeg(img_batch):
    d = []
    r = []
    for i in range(10, 105, 5):
        distortion, rate = run_jpeg(img_batch, i)
        d.append(distortion)
        r.append(rate)    
    plot_rate_distortion(d_jpeg=d, r_jpeg=r)

def run_jpeg(img_batch, quality=85):
    jpeg_buffer_batch = encode_jpeg(img_batch, quality)
    decoded_img_batch = decode_jpeg(jpeg_buffer_batch)

    distortion = calc_distortion(decoded_img_batch, img_batch)
    rate = calc_rate(jpeg_buffer_batch)

    return distortion, rate

def run_model(model, img_batch):
    model.eval()
    batch_size = img_batch.size(0)
    
    # Flatten and convert the image to a batch for the model
    img_batch = img_batch.view(batch_size, -1) # Flatten the imgs
    
    # Run the model
    with torch.no_grad():
        output, quantized = model(img_batch)

    # Unflatten the input and output to visualize it as an image
    img_batch = img_batch.view(batch_size, 1, 28, 28)
    output = output.view(batch_size, 1, 28, 28)

    # Calculate distortion (MSE)
    distortion = calc_distortion(img_batch, output)
    rate = calc_rate(quantized)

    return distortion, rate

def plot_rate_distortion(d_model=None, r_model=None, d_jpeg=None, r_jpeg=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if d_jpeg is not None and r_jpeg is not None:
        plt.plot(d_jpeg, r_jpeg, label='JPEG')
    if d_model is not None and r_model is not None:
        plt.plot(d_model, r_model, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if d_jpeg is not None and r_jpeg is not None:
        plt.scatter(d_jpeg, r_jpeg, label='JPEG')
    if d_model is not None and r_model is not None:
        plt.scatter(d_model, r_model, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Scatter Plot')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plots/rate_distortion_comparison.png')
    plt.show()

# Compare the rate-distortion of the model and JPEG
def run_test(model, test_loader, device, compare=False):
    d_jpeg = None
    d_model = None
    if compare:
        d_jpeg = []
        d_model = []
    r_jpeg = []
    r_model = []

    for img, _ in test_loader:
        img = img.to(device)
        if compare:
            distortion_jpeg, bit_rate_jpeg = run_jpeg(img)
            d_jpeg.append(distortion_jpeg)
            d_model.append(distortion_model)
        with torch.no_grad():   
            distortion_model, bit_rate_model = run_model(model, img)
        r_jpeg.append(bit_rate_jpeg)
        r_model.append(bit_rate_model)
        
    plot_rate_distortion(d_model=d_model, r_model=r_model, d_jpeg=d_jpeg, r_jpeg=r_jpeg)

def main():
    param_path = "../params/quantizer_MNIST_params_0.pth"
    model = MNIST_Coder()
    model, device = device_manager(model)
    model.load_state_dict(torch.load(param_path, map_location=device))
    model.eval()
    
    batch_size = 16
    test_loader = DataLoader(datasets.MNIST('../data/mnist',
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([transforms.ToTensor()])),
                            batch_size=batch_size,
                            shuffle=False)
    
    run_test(model, test_loader, device)

if __name__ == "__main__":
    main()