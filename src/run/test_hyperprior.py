import argparse
import random
import shutil
import sys
from models import ScaleHyperprior

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
from torchvision.io import encode_jpeg, decode_jpeg

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

from dataset import CustomNIHDataset
from utils import *

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
import pynvml

def test_epoch(test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

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

def run_qualitative_test(model, test_loader, device):
    img = test_loader.dataset[0]
    img = img.unsqueeze(1)
    img = img.to(device)
    
    # Run the model
    with torch.no_grad():
        model.eval()
        output_batch = model(img)["x_hat"]
        print(output_batch)
        # print(output_batch.shape)
        # distortion = calc_distortion(img, output_batch)
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

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

# Main function
def main(argv):
    args = parse_args(argv)

    pynvml.nvmlInit()
    free_memory = get_free_memory()
    best_gpu = free_memory.index(max(free_memory))

    # Set this GPU for PyTorch
    torch.cuda.set_device(best_gpu)
    print(f"Using GPU: {best_gpu}, Free Memory: {free_memory[best_gpu]}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = device_manager()
    test_transforms = transforms.Compose([transforms.Pad(padding=(16, 16, 16, 16), fill=0, padding_mode='constant'), transforms.ToTensor()])

    dataset_path = "/data/user3/data-resized/NIH/images-224"
    test_dataset = CustomNIHDataset(dataset_path, transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))
    lambd_ = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
    for i, (_l) in enumerate(lambd_):
        if i< 5:
            N = 128
            M = 192
        else:
            N = 192
            M = 320
        model = ScaleHyperprior(N,M)
        # model, device = device_manager(model)
        model = model.to(device)
    
        criterion = RateDistortionLoss(lmbda=_l)
        print(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    
        run_qualitative_test(model, test_loader, device)
        loss = test_epoch(test_loader, model, criterion)
        print(f"Loss: {loss}")
        # gen_RD_graph_for_jpeg(test_loader)
        # run_test(model, test_loader, lambda_, device, True)
        # run_test_with_lloyd(model, test_loader, lambda_, device)
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
