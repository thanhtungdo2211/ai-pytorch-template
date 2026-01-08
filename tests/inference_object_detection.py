import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.model as module_arch

# COCO class names
# WARNING: Verify if your model was trained with 0=Background or 0=Aeroplane.
# Standard FasterRCNN expects 0=Background.
COCO_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def simple_object_detection(image_path, checkpoint_path='saved/models/ObjectDetection_FasterRCNN/model_best.pth',
                            confidence_threshold=0.5, device=None):
    
    # 1. Setup Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # 2. Initialize model
    # Ensure num_classes matches your checkpoint (e.g. 21)
    model = module_arch.FasterRCNN(num_classes=len(COCO_CLASSES), pretrained=False)
    model.to(device)
    
    # 3. Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    # weights_only=False is needed for some older pytorch saves, but True is safer if possible
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 4. Preprocessing
    # Note: If you trained on 416x416, you might want to resize here, 
    # but FasterRCNN handles variable sizes internally.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 5. Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 6. Inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)
    print(f"Inference time: {time.time() - start_time:.3f}s")
    
    # 7. Extract detections
    # Move back to CPU for numpy processing
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # 8. Filter by confidence threshold
    keep_idx = scores >= confidence_threshold
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Number of objects detected: {len(boxes)}")
    print(f"{'='*70}")
    
    detections = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Safety check for label index
        if label < len(COCO_CLASSES):
            class_name = COCO_CLASSES[label]
        else:
            class_name = f"Unknown Class {label}"
            
        print(f"{i+1}. {class_name:15s} | Confidence: {score*100:5.2f}% | Box: {box.astype(int)}")
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'box': box.tolist()
        })
    
    print(f"{'='*70}\n")
    
    return detections, image, boxes, scores, labels

def visualize_detections(image_path, checkpoint_path, confidence_threshold=0.5, output_path='detection_result.jpg'):
    detections, image, boxes, scores, labels = simple_object_detection(
        image_path, checkpoint_path, confidence_threshold
    )
    
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image)
    # Convert RGB to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        
        if label < len(COCO_CLASSES):
            class_name = COCO_CLASSES[label]
        else:
            class_name = f"Class {label}"
        
        # Draw box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background for better visibility
        label_text = f"{class_name}: {score*100:.1f}%"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_array, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(img_array, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, img_array)
    print(f"Result saved to: {output_path}")
    return detections

if __name__ == '__main__':
    # Example usage
    image_path = '/mnt/d/tungdt/dataset/1/images/1.jpeg'
    checkpoint_path = '/mnt/d/tungdt/ai-pytorch-template/saved/models/ObjectDetection_FasterRCNN/1021_085953/model_best.pth'
    
    if os.path.exists(image_path) and os.path.exists(checkpoint_path):
        visualize_detections(image_path, checkpoint_path, confidence_threshold=0.01)
    else:
        print("Please check your image and checkpoint paths.")