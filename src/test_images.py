import argparse
import torch
import matplotlib.pyplot as plt
from model import MNIST_Coder
from torchvision import transforms, datasets
import cv2
import os
import numpy as np
import torchvision
from utils import device_manager

def calculate_mse(image_a, image_b):
    # Ensure the images are floats
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)
    mse = np.mean((image_a - image_b) ** 2)
    return mse

def run_jpeg(image_path, quality=85):
    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Save the image with JPEG compression
    jpeg_path = '../data/tmp/compressed_image.jpeg'
    cv2.imwrite(jpeg_path, original_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Load the JPEG image back
    jpeg_image = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)

    # Calculate distortion (MSE) and bit rate (file size)
    distortion = calculate_mse(original_image, jpeg_image)
    file_size = os.path.getsize(jpeg_path) * 8  # Bit rate approximated by file size in bits

    return distortion, file_size

def run_model(model, img):
    # Flatten and convert the image to a batch for the model
    img_batch = img.unsqueeze(0).view(1, -1)
    
    # Run the model
    with torch.no_grad():
        # Compress and then decompress the image
        compressed_data = model.compress(img_batch)
        output_batch = model.decompress(compressed_data)

    # Unflatten the input and output to visualize it as an image
    original_img_np = img.squeeze().numpy()
    output_img_np = output_batch.view(1, 28, 28).squeeze().numpy()

    # Calculate distortion (MSE)
    distortion = calculate_mse(original_img_np * 255, output_img_np * 255)  # Scale images back to 0-255

    # Calculate rate (size of the compressed representation in bits)
    # Note: You need to ensure that 'compressed_data' is a bytes object for this to work correctly
    file_size = len(compressed_data) * 8  # Convert byte size to bit size

    return distortion, file_size

# Compare the rate-distortion of the model and JPEG
def compare_rate_distortion(model, test_set):
    distortions_jpeg = []
    distortions_model = []
    bit_rates_jpeg = []
    bit_rates_model = []

    for i in range(10):
        img, _ = test_set[i]
        # Assuming you save each test image to compare with JPEG
        image_path = f'../data/tmp/test_image_{i}.png'
        torchvision.utils.save_image(img, image_path)
        
        distortion_jpeg, bit_rate_jpeg = run_jpeg(image_path)
        distortion_model, bit_rate_model = run_model(model, img)

        distortions_jpeg.append(distortion_jpeg)
        distortions_model.append(distortion_model)
        bit_rates_jpeg.append(bit_rate_jpeg)
        bit_rates_model.append(bit_rate_model)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(distortions_jpeg, bit_rates_jpeg, label='JPEG')
    plt.plot(distortions_model, bit_rates_model, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(distortions_jpeg, bit_rates_jpeg, label='JPEG')
    plt.scatter(distortions_model, bit_rates_jpeg, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Scatter Plot')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plots/rate_distortion_comparison.png')
    plt.show()

def main():
    param_path = "../params/quantizer_MNIST_params_0.pth"
    model = MNIST_Coder()
    model, device = device_manager(model)
    model.load_state_dict(torch.load(param_path, map_location=device))
    model.eval()
    
    test_transform = transforms.Compose([transforms.ToTensor()]) 
    test_set = datasets.MNIST('../data/mnist', train=False, download=True, transform=test_transform)
    
    compare_rate_distortion(model, test_set)

if __name__ == "__main__":
    main()