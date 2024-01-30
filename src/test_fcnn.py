import argparse
import torch
import matplotlib.pyplot as plt
from model import MNIST_FCNN
from torchvision import transforms, datasets
from torchvision.io import encode_jpeg, decode_jpeg
from utils import device_manager, calc_distortion, calc_empirical_rate
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
    rate = calc_empirical_rate(jpeg_buffer_batch)

    return distortion, rate

def run_model(model, img_batch):
    model.eval()
    
    # Run the model
    with torch.no_grad():
        output, quantized = model(img_batch)

    # Calculate distortion (MSE)
    distortion = calc_distortion(img_batch, output)
    rate = calc_empirical_rate(quantized)

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

def run_qualitative_test(model, test_loader, lambda_, device):
    for l_ in lambda_:
        
        img, _ = test_loader.dataset[0]
        img = img.to(device)
        
        # Run the model
        with torch.no_grad():
            model.load_state_dict(torch.load(f"../params/fcnn_params_{l_}.pth", map_location=device))
            model.eval()
            output_batch, _ = model(img)
            distortion = calc_distortion(img, output_batch)
        print(f"Distortion: {distortion}")
        
        # Unflatten the input and output to visualize it as an image
        img = img.squeeze().cpu().numpy()
        output_img = output_batch.squeeze().cpu().numpy()
        
        f = plt.figure() 
        f.add_subplot(1,2,1) 
        plt.imshow(img, cmap='gray') 
        f.add_subplot(1,2,2) 
        plt.imshow(output_img, cmap='gray') 
        plt.show() 

# Compare the rate-distortion of the model and JPEG
def run_test(model, test_loader, lambda_, device, compare=False):
    d_jpeg = None
    r_jpeg = None
    if compare:
        d_jpeg = []
        r_jpeg = []
    d_model = []
    r_model = []
    for l_ in lambda_:
        model.load_state_dict(torch.load(f"../params/fcnn_params_{l_}.pth", map_location=device))
        model.eval()
        running_r_jpeg=0
        running_d_jpeg=0
        running_r_model=0
        running_d_model=0
        
        for img, _ in test_loader:
            img = img.to(device)
            if compare:
                distortion_jpeg, rate_jpeg = run_jpeg(img)
                running_r_jpeg+=rate_jpeg
                running_d_jpeg+=distortion_jpeg
            with torch.no_grad():   
                distortion_model, rate_model = run_model(model, img)
                running_r_model+=rate_model
                running_d_model+=distortion_model
        
        if compare:
            j_avg_rate = running_r_jpeg/len(test_loader)
            j_avg_distortion = running_d_jpeg/len(test_loader)
            print(f"JPEG: Rate: {j_avg_rate}, Distortion: {j_avg_distortion}")
            r_jpeg.append(j_avg_rate)
            d_jpeg.append(j_avg_distortion)
            
        r_avg_rate = running_r_model/len(test_loader)
        r_avg_distortion = running_d_model/len(test_loader)
        print(f"Model: Rate: {r_avg_rate}, Distortion: {r_avg_distortion}")
        
        r_model.append(r_avg_rate.cpu().detach().numpy())
        d_model.append(r_avg_distortion.cpu().detach().numpy())
        
        
    plot_rate_distortion(d_model=d_model, r_model=r_model, d_jpeg=d_jpeg, r_jpeg=r_jpeg)

# Main function
def main():
    model = MNIST_FCNN()
    model, device = device_manager(model)
    
    # lambda_ = [0.001]
    lambda_ = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
    batch_size = 256
    test_loader = DataLoader(datasets.MNIST('../data/mnist',
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([transforms.ToTensor()])),
                            batch_size=batch_size,
                            shuffle=False)
    
    run_test(model, test_loader, lambda_, device)
    # run_qualitative_test(model, test_loader, lambda_, device)

if __name__ == "__main__":
    main()