import argparse
import torch
import matplotlib.pyplot as plt
from model import Quantizer_Images
from torchvision import transforms, datasets

def run_test(model, test_set):
    img, _ = test_set[0]
    
    img_batch = img.unsqueeze(0).view(1, -1)
    
    # Run the model
    with torch.no_grad():
        output_batch, _ = model(img_batch)
    
    # Unflatten the input and output to visualize it as an image
    img = img.squeeze().numpy()
    output_img = output_batch.view(1, 28, 28).squeeze().numpy()
    
    f = plt.figure() 
    f.add_subplot(1,2,1) 
    plt.imshow(img, cmap='gray') 
    f.add_subplot(1,2,2) 
    plt.imshow(output_img, cmap='gray') 
    plt.show() 
    
def main():
    nums = [0,1,2,3]
    for num in nums:
        model = Quantizer_Images()
        model.load_state_dict(torch.load(f"../params/quantizer_MNIST_params_{num}.pth"))
        model.eval()
        test_transform = transforms.Compose([transforms.ToTensor()]) 
        test_set = datasets.MNIST('../data/mnist', train=False, download=False, transform=test_transform)
        
        run_test(model, test_set)

if __name__ == "__main__":

    # Call the main function with parsed arguments
    main()