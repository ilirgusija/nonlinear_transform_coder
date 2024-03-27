import argparse
import random
import shutil
import sys
from compressai.models import ScaleHyperprior

import torch
import torch.nn as nn
import torch.optim as optim

from utils import device_manager

import torchxrayvision as xrv
from torchxrayvision.datasets import NIH_Dataset

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models


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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

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
        d = data['img']
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
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

            aux_loss.update(model.aux_loss())
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

def save_checkpoint(state, is_best, filename="/data/user3/params/checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")

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
        default=4,
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = device_manager()
    train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    # test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    dataset_path = "/data/user3/data-resized/NIH/images-224"
    train_dataset = NIH_Dataset(dataset_path)
    # test_dataset = NIH_Dataset(dataset_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"))
    print(train_loader)
    print(type(train_loader))
    print(len(train_loader))
    # test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))
    # lambd_ = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
    lambd_ = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
    for i, (_l) in enumerate(lambd_):
        if i< 5:
            N = 128
            M = 192
        else:
            N = 192
            M = 320
        model = ScaleHyperprior(N,M)

        if device=='cuda' and torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)

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
                )


if __name__ == "__main__":
    main(sys.argv[1:])
