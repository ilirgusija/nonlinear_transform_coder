import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Optional, List

class CustomNIHDataset(Dataset):
    def __init__(self, img_dir: str, dataset_type: str = "train", transform: Optional[Callable] = None):
        """
        Args:
            img_dir (str): Directory with all the NIH images.
            dataset_type (str): Type of the dataset to return ("train", "validation", "test").
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.image_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir) if img_name.endswith('.png') or img_name.endswith('.jpg')]
        
        # Partition the dataset
        self._partition_dataset()

    def _partition_dataset(self):
        # Split the dataset into training (70%), validation (15%), and testing (15%)
        train_paths, temp_paths = train_test_split(self.image_paths, test_size=0.3, random_state=42)  # Split for training
        validation_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)  # Split remaining for validation and test
        
        # Assign the appropriate paths based on the dataset type
        if self.dataset_type == "train":
            self.image_paths = train_paths
        elif self.dataset_type == "validation":
            self.image_paths = validation_paths
        elif self.dataset_type == "test":
            self.image_paths = test_paths
        else:
            raise ValueError("Invalid dataset type specified. Choose from 'train', 'validation', or 'test'.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            img = self.transform(img)

        return img

