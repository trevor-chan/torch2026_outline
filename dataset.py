import torch
from torchvision import datasets
from torch.utils.data import IterableDataset
import random
from image_utils import flatten_image, normalize_image, one_hot_encode
from torchvision import datasets, transforms

class MNISTDataset(IterableDataset):
    """
    A simplified MNIST dataset that yields samples infinitely.
    Provides raw data without transforms, using a random seed to control shuffling.
    """
    def __init__(self, data_dir='./data', train=True, seed=0):
        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        
        # Download and load the training data
        if self.train:
            self.dataset = datasets.MNIST(
                root=self.data_dir, train=self.train, download=True, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transforms.ToTensor())
        
        # Set up randomization
        self.indices = list(range(len(self.dataset)))
        random.seed(self.seed)
        
        print(f"Dataset initialized with {'training' if train else 'test'} data, {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """
        Create an iterator that yields samples one by one indefinitely.
        Uses the random seed for reproducibility in shuffling.
        """
        # Create a copy of indices and shuffle them
        indices = self.indices.copy()
        random.shuffle(indices)
        
        position = 0
        
        while True:
            # Reshuffle when we've gone through all samples
            if position >= len(indices):
                random.shuffle(indices)
                position = 0
            
            # Get the current sample
            idx = indices[position]
            image, label = self.dataset[idx]
            
            # Convert image to tensor if it's not already
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            
            # Normalize and flatten the image using utility functions
            image = normalize_image(image)
            image = flatten_image(image)
            
            label = one_hot_encode(label)
            
            # Move to next position
            position += 1
            
            yield image, label 