import torch
import sys
import matplotlib.pyplot as plt
from model import MNIST_FCNN
from torchvision import transforms, datasets
from torchvision.io import encode_jpeg, decode_jpeg
from utils import device_manager, calc_distortion, calc_empirical_rate
from torch.utils.data import DataLoader

# Generate the Rate Distortion graph for JPEG based on quality
def gen_RD_vals_for_jpeg(test_loader, device):
    r_list = []
    d_list = []
    for i in range(10, 105, 5):
        running_r=0
        running_d=0
        for img, _ in test_loader:
            with torch.no_grad():   
                distortion_jpeg, rate_jpeg = run_jpeg(img, device, i)
                running_r+=rate_jpeg
                running_d+=distortion_jpeg
        r_avg_rate = running_r/len(test_loader)
        r_avg_distortion = running_d/len(test_loader)
        print(f"Model: Rate: {r_avg_rate}, Distortion: {r_avg_distortion}")
        
        r_list.append(r_avg_rate.cpu().detach().numpy())
        d_list.append(r_avg_distortion.cpu().detach().numpy())
    return r_list, d_list

def run_jpeg(img_batch, device, quality=85):
    batch_size = img_batch.size(0)
    distortions = []
    rates = []
    
    for idx in range(batch_size):
        img = (img_batch[idx] * 255).to(torch.uint8)  # Convert each image in the batch
        jpeg_buffer = encode_jpeg(img, quality=quality)
        decoded_img = decode_jpeg(jpeg_buffer).to(device)
        print(jpeg_buffer)
        sys.exit()
        # Ensure decoded_img and original img are in the same dtype and range for comparison
        decoded_img = decoded_img.float() / 255
        
        distortion = calc_distortion(decoded_img.unsqueeze(0), img_batch[idx].unsqueeze(0).to(device))
        rate = calc_empirical_rate(jpeg_buffer)  # Ensure this reflects the byte size
        
        distortions.append(distortion)
        rates.append(rate)

    # Calculate mean distortion and rate for the batch
    mean_distortion = torch.mean(torch.stack(distortions))
    mean_rate = torch.mean(torch.stack(rates))

    return mean_distortion, mean_rate

def run_model(model, img_batch):
    model.eval()
    
    # Run the model
    with torch.no_grad():
        output, quantized = model(img_batch)

    # Calculate distortion (MSE)
    distortion = calc_distortion(img_batch, output)
    rate = calc_empirical_rate(quantized)

    return distortion, rate

def plot_rate_distortion(d_model=None, r_model=None, d_jpeg=None, r_jpeg=None, img_name="model"):
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
    plt.savefig(f"../plots/rate_distortion_comparison_{img_name}.png")
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
        # print(f"Distortion: {distortion}")
        
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
    print("Running JPEG=================")
    r_jpeg, d_jpeg = gen_RD_vals_for_jpeg(test_loader, 'cpu')    
    print("Running Model=============")
    d_model = []
    r_model = []
    for l_ in lambda_:
        model.load_state_dict(torch.load(f"../params/fcnn_params_{l_}.pth", map_location=device))
        model.eval()
        running_r_model=0
        running_d_model=0
        
        for img, _ in test_loader:
            img = img.to(device)
            with torch.no_grad():   
                distortion_model, rate_model = run_model(model, img)
                running_r_model+=rate_model
                running_d_model+=distortion_model
            
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
    
    lambda_ = [0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
    batch_size = 1
    test_loader = DataLoader(datasets.MNIST('../data/mnist',
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=batch_size,
                             shuffle=False)
    
    # gen_RD_graph_for_jpeg(test_loader)
    # run_test(model, test_loader, lambda_, device, True)
    # run_test_with_lloyd(model, test_loader, lambda_, device)
    run_qualitative_test(model, test_loader, lambda_, device)

if __name__ == "__main__":
    main()
