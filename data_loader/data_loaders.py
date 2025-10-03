from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # CIFAR-10 normalization values
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        
        if training:
            # Data augmentation for training
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # No augmentation for validation/test
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(
            root=self.data_dir, 
            train=training, 
            download=True, 
            transform=transform
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class DummyDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        
        if training:
            # Data augmentation for training
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # No augmentation for validation/test
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(
            root=self.data_dir, 
            train=training, 
            download=True, 
            transform=transform
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)