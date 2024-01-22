import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

class NIHChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

    def __len__(self):
        if self.split == 'train':
            return 100000
        elif self.split == 'val':
            return 10000
        elif self.split == 'test':
            return 10000

    def __getitem__(self, idx):
        if self.split == 'train':
            image_path = os.path.join(self.root_dir, 'train', f'{idx}.png')
        elif self.split == 'val':
            image_path = os.path.join(self.root_dir, 'val', f'{idx}.png')
        elif self.split == 'test':
            image_path = os.path.join(self.root_dir, 'test', f'{idx}.png')

        image = Image.open(image_path)
        image = ToTensor()(image)

        return image

if __name__ == '__main__':
    dataset = NIHChestXrayDataset('/path/to/dataset')
    print(len(dataset))
    image, label = dataset[0]
    print(image.size())
    print(label)
