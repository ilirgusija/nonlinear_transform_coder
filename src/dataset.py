import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from torchxrayvision.datasets import CheX_Dataset

if __name__ == '__main__':
    dataset = CheX_Dataset("/data/user3_data/chexpertchestxrays-u20210408/", csvpath="/data/user3_data/chexpertchestxrays-u20210408/train_visualCheXbert.csv")
    print(len(dataset))
    image, label = dataset[0]
    print(image.size())
    print(label)
