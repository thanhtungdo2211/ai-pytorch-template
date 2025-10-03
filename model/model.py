import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGGModel(BaseModel):
    def __init__(self, num_classes=10, vgg_type='VGG16'):
        super().__init__()
        
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        self.features = self._make_layers(cfg=cfg[vgg_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # RGB images
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class YOLOv3(BaseModel):
    def __init__(self, num_classes=80, img_size=416):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Darknet-53 backbone (simplified)
        self.backbone = self._create_backbone()
        
        # Detection heads for 3 different scales
        self.head_1 = self._create_detection_head(1024, num_classes)
        self.head_2 = self._create_detection_head(512, num_classes)
        self.head_3 = self._create_detection_head(256, num_classes)
        
    def _create_backbone(self):
        # Simplified backbone using ResNet
        resnet = models.resnet50(pretrained=True)
        layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        return nn.Sequential(*layers)
    
    def _create_detection_head(self, in_channels, num_classes):
        # Each grid cell predicts 3 bounding boxes
        # Each box: [x, y, w, h, conf, class_probs...]
        anchors_per_cell = 3
        output_size = anchors_per_cell * (5 + num_classes)
        
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, output_size, 1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply detection heads (simplified - normally you'd have FPN)
        detection_1 = self.head_1(features)
        
        return {
            'predictions': detection_1,
            'features': features
        }


class FasterRCNN(BaseModel):
    def __init__(self, num_classes=91, pretrained=True):
        super().__init__()
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classifier head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)


class SimpleObjectDetector(BaseModel):
    """Simplified single-shot detector for learning purposes"""
    def __init__(self, num_classes=20, backbone='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Detection head
        # Predicts: [x, y, w, h, objectness, class_probs...]
        self.detection_head = nn.Sequential(
            nn.Conv2d(backbone_out, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + num_classes, 1)  # 4 box coords + 1 objectness + classes
        )
        
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections