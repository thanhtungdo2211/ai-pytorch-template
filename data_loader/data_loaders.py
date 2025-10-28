import os
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image

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
        
class COCODetectionDataLoader(BaseDataLoader):
    """
    COCO-style detection data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, image_size=416):
        
        self.image_size = image_size
        
        # Define transforms
        if training:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.dataset = DetectionDataset(data_dir, transform=transform, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=detection_collate_fn)


class YOLODataset(Dataset):
    """
    Dataset for YOLO format annotations (.txt files).
    Each label file should contain lines with: <class> <x_center> <y_center> <width> <height>
    where coordinates are normalized (0..1) relative to image width/height.
    """
    def __init__(self, data_dir, transform=None, training=True, images_subdir='images', labels_subdir='labels'):
        self.data_dir = data_dir
        self.transform = transform
        self.training = training

        self.images = []
        self.labels = []

        images_dir = os.path.join(self.data_dir, images_subdir)
        labels_dir = os.path.join(self.data_dir, labels_subdir)

        # If images folder doesn't exist, try root
        if not os.path.exists(images_dir):
            images_dir = self.data_dir

        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                self.images.append(img_path)

                # corresponding label path
                label_name = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)

                # if labels dir doesn't exist or file not found, check same dir as image
                if not os.path.exists(label_path):
                    candidate = os.path.join(images_dir, label_name)
                    label_path = candidate if os.path.exists(candidate) else None

                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        boxes = []
        labels = []

        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        x_c = float(parts[1]) * w
                        y_c = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h

                        xmin = x_c - bw / 2.0
                        ymin = y_c - bh / 2.0
                        xmax = x_c + bw / 2.0
                        ymax = y_c + bh / 2.0

                        # clip to image
                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = min(w, xmax)
                        ymax = min(h, ymax)

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls)
                    except ValueError:
                        # skip malformed lines
                        continue

        if len(boxes) == 0:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }

        if self.transform:
            image = self.transform(image)

        return image, target


class YOLODetectionDataLoader(BaseDataLoader):
    """
    DataLoader wrapper for YOLO-format datasets.
    Expects dataset structure like:
      data_dir/
        images/
          xxx.jpg
        labels/
          xxx.txt
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, image_size=416):

        self.image_size = image_size

        if training:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.dataset = YOLODataset(data_dir, transform=transform, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=detection_collate_fn)


class VOCDetectionDataLoader(BaseDataLoader):
    """
    Pascal VOC detection data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, image_size=416):
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Use torchvision's VOC dataset
        self.dataset = datasets.VOCDetection(
            root=data_dir,
            year='2012',
            image_set='train' if training else 'val',
            download=False,
            transform=transform,
            target_transform=self._parse_voc_target
        )
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=detection_collate_fn)
    
    def _parse_voc_target(self, target):
        """Parse VOC XML annotation"""
        boxes = []
        labels = []
        
        root = target['annotation']
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Map class name to label (you'd need a proper class mapping)
            class_name = obj.find('name').text
            label = self._class_to_idx(class_name)
            labels.append(label)
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
    
    def _class_to_idx(self, class_name):
        # VOC classes
        voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor']
        return voc_classes.index(class_name) if class_name in voc_classes else 0


class DetectionDataset(Dataset):
    """
    Custom detection dataset
    """
    def __init__(self, data_dir, transform=None, training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.training = training
        
        # Load image and annotation file lists
        self.images = []
        self.annotations = []
        self._load_data()
    
    def _load_data(self):
        images_dir = os.path.join(self.data_dir, 'images')
        annotations_dir = os.path.join(self.data_dir, 'annotations')
        
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(images_dir, img_file))
                    
                    # Corresponding annotation file
                    ann_file = img_file.replace('.jpg', '.json').replace('.png', '.json')
                    ann_path = os.path.join(annotations_dir, ann_file)
                    self.annotations.append(ann_path if os.path.exists(ann_path) else None)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation
        ann_path = self.annotations[idx]
        if ann_path and os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            target = self._parse_annotation(annotation)
        else:
            # Dummy target for images without annotations
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _parse_annotation(self, annotation):
        """Parse custom annotation format"""
        boxes = []
        labels = []
        
        for obj in annotation.get('objects', []):
            bbox = obj['bbox']  # Assume [x, y, w, h] format
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(obj['class_id'])
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }


def detection_collate_fn(batch):
    """
    Custom collate function for detection data
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets