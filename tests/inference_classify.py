import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.model as module_arch


sys.modules["pathlib._local"] = pathlib 
if os.name == 'nt':
   pathlib.PosixPath = pathlib.WindowsPath
else:
   pathlib.WindowsPath = pathlib.PosixPath


# Class names for Weather dataset
WEATHER_CLASSES = [
    'dew', 'fogsmog', 'frost', 'glaze', 'hail', 
    'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow'
]

def simple_inference(image_path, checkpoint_path='saved/models/Weather_VGG16/1028_090401/model_best.pth'):
    """
    Simple inference on a single weather image
    
    Args:
        image_path: Path to JPG image
        checkpoint_path: Path to model checkpoint
    """
    
    # Initialize model
    model = module_arch.VGGModel(num_classes=11, vgg_type='VGG16')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted: {WEATHER_CLASSES[predicted_class]} ({confidence*100:.2f}%)")
    print(f"\nTop 5 predictions:")
    top5_prob, top5_idx = torch.topk(probabilities[0], k=5)
    for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
        print(f"  {i+1}. {WEATHER_CLASSES[idx]}: {prob.item()*100:.2f}%")
    print(f"{'='*60}\n")
    
    return WEATHER_CLASSES[predicted_class], confidence


if __name__ == '__main__':
    # Example usage
    image_path = '/mnt/d/tungdt/dataset/dataset/dew/2283.jpg'
    
    predicted_label, confidence = simple_inference(image_path)
    print(f"Result: {predicted_label} with {confidence*100:.2f}% confidence")