import argparse
import random
import shutil
import sys
from models import ScaleHyperprior
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from dataset import CustomNIHDataset

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
import pynvml

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def train_one_epoch(
    model, criterion, train_loader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, (data) in enumerate(train_loader):
        d = data
        d = d.to(device)
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss() if not hasattr(model, 'module') else model.module.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_loader.dataset)}"
                f" ({100. * i / len(train_loader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

def test_epoch(epoch, test_dataloader, model, criterion):
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

            aux_loss.update(model.aux_loss() if not hasattr(model, 'module') else model.module.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def save_checkpoint(state, is_best, _l, filename="/data/user3/params/checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"checkpoint_best_loss_{_l}.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
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

def main(argv):
    args = parse_args(argv)
        # Open the log file
    # Redirect stdout to the log file
    # pynvml.nvmlInit()
    # # Get the ID of the GPU with the most free memory
    # free_memory = get_free_memory()
    # best_gpu = free_memory.index(max(free_memory))
    # # Set this GPU for PyTorch
    # torch.cuda.set_device(best_gpu)
    # print(f"Using GPU: {best_gpu}, Free Memory: {free_memory[best_gpu]}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = device_manager()
    print(device)
    train_transforms = transforms.Compose([transforms.Pad(padding=(16, 16, 16, 16), fill=0, padding_mode='constant'), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Pad(padding=(16, 16, 16, 16), fill=0, padding_mode='constant'), transforms.ToTensor()])

    dataset_path = "/data/user3/data-resized/NIH/images-224"
    train_dataset = CustomNIHDataset(dataset_path, dataset_type="train", transform=train_transforms)
    test_dataset = CustomNIHDataset(dataset_path, dataset_type="validation", transform=test_transforms)
     
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    # lambd_ = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
    lambd_ = [0.1800]
    for i, (_l) in enumerate(lambd_):
        if i< 5:
            N = 128
            M = 192
        else:
            N = 192
            M = 320
        model = ScaleHyperprior(N,M)
        model, _ = device_manager(model)
        # model = model.to(device)

        optimizer, aux_optimizer = configure_optimizers(model, args)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        criterion = RateDistortionLoss(lmbda=_l)

        last_epoch = 0
        if args.checkpoint:  # load from previous checkpoint
            print("Loading", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        best_loss = float("inf")
        for epoch in range(last_epoch, args.epochs):
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
            )
            loss = test_epoch(epoch, test_dataloader, model, criterion)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    _l,
                    f"/data/user3/params/checkpoint_best_loss_{_l}.pth"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
