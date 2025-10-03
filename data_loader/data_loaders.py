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

class CustomImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, image_size=224):

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]   # ImageNet stds
        )

        if training:
                # Data augmentation for training
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
                
        else:
                # No augmentation for validation/test
                transform = transforms.Compose([
                    transforms.Resize(int(image_size * 1.14)),  # Resize to slightly larger
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            root=self.data_dir,
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