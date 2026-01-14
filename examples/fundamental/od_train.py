import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

class ObjectDetectionHead(nn.Module):
    """
    A simplified object detection head demonstrating dropout and batch normalization
    """
    def __init__(self, in_channels=2048, num_classes=80, num_anchors=9):
        super(ObjectDetectionHead, self).__init__()
        
        # Classification head with BatchNorm and Dropout
        self.cls_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_bn1 = nn.BatchNorm2d(256)
        self.cls_dropout1 = nn.Dropout2d(p=0.3)  # 2D dropout for conv layers
        
        self.cls_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cls_bn2 = nn.BatchNorm2d(128)
        self.cls_dropout2 = nn.Dropout2d(p=0.2)
        
        # Final classification layer
        self.cls_head = nn.Conv2d(128, num_anchors * num_classes, kernel_size=1)
        
        # Regression head (bounding box prediction)
        self.reg_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.reg_bn1 = nn.BatchNorm2d(256)
        self.reg_dropout1 = nn.Dropout2d(p=0.3)
        
        self.reg_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.reg_bn2 = nn.BatchNorm2d(128)
        self.reg_dropout2 = nn.Dropout2d(p=0.2)
        
        # Final regression layer (4 coordinates per anchor)
        self.reg_head = nn.Conv2d(128, num_anchors * 4, kernel_size=1)
        
        # Objectness head (binary classification: object vs background)
        self.obj_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.obj_bn1 = nn.BatchNorm2d(128)
        self.obj_dropout1 = nn.Dropout2d(p=0.2)
        
        self.obj_head = nn.Conv2d(128, num_anchors, kernel_size=1)
        
    def forward(self, x):
        # Classification branch
        cls = F.relu(self.cls_bn1(self.cls_conv1(x)))
        cls = self.cls_dropout1(cls)
        cls = F.relu(self.cls_bn2(self.cls_conv2(cls)))
        cls = self.cls_dropout2(cls)
        cls_output = self.cls_head(cls)
        
        # Regression branch
        reg = F.relu(self.reg_bn1(self.reg_conv1(x)))
        reg = self.reg_dropout1(reg)
        reg = F.relu(self.reg_bn2(self.reg_conv2(reg)))
        reg = self.reg_dropout2(reg)
        reg_output = self.reg_head(reg)
        
        # Objectness branch
        obj = F.relu(self.obj_bn1(self.obj_conv1(x)))
        obj = self.obj_dropout1(obj)
        obj_output = self.obj_head(obj)
        
        return {
            'classification': cls_output,
            'regression': reg_output,
            'objectness': obj_output
        }

class SimpleObjectDetector(nn.Module):
    """
    Complete object detection model using ResNet backbone with detection head
    """
    def __init__(self, num_classes=80, pretrained=True):
        super(SimpleObjectDetector, self).__init__()
        
        # Backbone: ResNet50 (already includes BatchNorm layers)
        backbone = resnet50(pretrained=pretrained)
        
        # Remove final avgpool and fc layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Feature Pyramid Network-like structure with BatchNorm
        self.fpn_conv = nn.Conv2d(2048, 256, kernel_size=1)
        self.fpn_bn = nn.BatchNorm2d(256)
        
        # Detection head
        self.detection_head = ObjectDetectionHead(
            in_channels=256, 
            num_classes=num_classes
        )
        
        # Additional fully connected layers with regular dropout
        self.fc_features = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_dropout = nn.Dropout(p=0.5)  # Regular dropout for FC layers
        
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply FPN-like conv
        fpn_features = F.relu(self.fpn_bn(self.fpn_conv(features)))
        
        # Detection head outputs
        detection_outputs = self.detection_head(fpn_features)
        
        # Additional classification branch (global features)
        global_features = self.fc_features(fpn_features)
        global_features = global_features.view(global_features.size(0), -1)
        
        fc_out = F.relu(self.fc1_bn(self.fc1(global_features)))
        fc_out = self.fc1_dropout(fc_out)
        global_cls = self.fc2(fc_out)
        
        return {
            'detection': detection_outputs,
            'global_classification': global_cls,
            'features': fpn_features
        }

# Example usage and training setup
def train_example():
    # Create model
    model = SimpleObjectDetector(num_classes=20)  # VOC dataset has 20 classes
    
    # Set model to training mode (important for BatchNorm and Dropout)
    model.train()
    
    # Example input (batch_size=4, channels=3, height=416, width=416)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 416, 416)
    
    # Forward pass
    with torch.set_grad_enabled(True):  # Enable gradients for training
        outputs = model(input_tensor)
        
        # Extract outputs
        detection_outputs = outputs['detection']
        cls_logits = detection_outputs['classification']
        bbox_preds = detection_outputs['regression'] 
        objectness = detection_outputs['objectness']
        global_cls = outputs['global_classification']
        
        print("Output shapes:")
        print(f"Classification logits: {cls_logits.shape}")
        print(f"Bounding box predictions: {bbox_preds.shape}")
        print(f"Objectness scores: {objectness.shape}")
        print(f"Global classification: {global_cls.shape}")

def inference_example():
    # Create model
    model = SimpleObjectDetector(num_classes=20)
    
    # Set model to evaluation mode (important for BatchNorm and Dropout)
    model.eval()
    
    # Example input for inference
    input_tensor = torch.randn(1, 3, 416, 416)
    
    # Forward pass without gradients
    with torch.no_grad():
        outputs = model(input_tensor)
        
        detection_outputs = outputs['detection']
        cls_logits = detection_outputs['classification']
        bbox_preds = detection_outputs['regression']
        objectness = detection_outputs['objectness']
        
        # Apply sigmoid to objectness and softmax to classification
        objectness_probs = torch.sigmoid(objectness)
        cls_probs = torch.softmax(cls_logits.view(-1, 20), dim=1)
        
        print("Inference results:")
        print(f"Objectness probabilities shape: {objectness_probs.shape}")
        print(f"Classification probabilities shape: {cls_probs.shape}")

# Demonstration of BatchNorm and Dropout behavior
def demonstrate_train_vs_eval():
    model = SimpleObjectDetector(num_classes=20)
    input_tensor = torch.randn(2, 3, 416, 416)
    
    # Training mode
    model.train()
    print("=== TRAINING MODE ===")
    print("- Dropout: Active (randomly drops neurons)")
    print("- BatchNorm: Uses batch statistics")
    
    with torch.no_grad():
        train_output1 = model(input_tensor)
        train_output2 = model(input_tensor)  # Same input, different output due to dropout
    
    print(f"Same input, different outputs due to dropout:")
    print(f"Output 1 sum: {train_output1['global_classification'].sum().item():.4f}")
    print(f"Output 2 sum: {train_output2['global_classification'].sum().item():.4f}")
    
    # Evaluation mode
    model.eval()
    print("\n=== EVALUATION MODE ===")
    print("- Dropout: Inactive (all neurons active)")
    print("- BatchNorm: Uses running statistics")
    
    with torch.no_grad():
        eval_output1 = model(input_tensor)
        eval_output2 = model(input_tensor)  # Same input, same output
    
    print(f"Same input, same outputs (no dropout):")
    print(f"Output 1 sum: {eval_output1['global_classification'].sum().item():.4f}")
    print(f"Output 2 sum: {eval_output2['global_classification'].sum().item():.4f}")

if __name__ == "__main__":
    print("Object Detection Model with Dropout and BatchNorm\n")
    
    # Run examples
    train_example()
    print("\n" + "="*50 + "\n")
    
    inference_example()
    print("\n" + "="*50 + "\n")
    
    demonstrate_train_vs_eval()