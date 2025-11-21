import torch
import os
import sys
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def main():
    # 1. Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 2. Load Model Structure
    num_classes = 2  # background + person
    model = get_model_instance_segmentation(num_classes)
    
    # 3. Load Trained Weights
    model_path = "model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 4. Load and Preprocess Image
    image_path = "data/PennFudanPed/PNGImages/FudanPed00046.png"
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
        
    print(f"Processing image: {image_path}")
    image = read_image(image_path)
    eval_transform = get_transform(train=False)

    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # 5. Visualize Results
    print("Visualizing results...")
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    
    # Filter low confidence predictions
    score_threshold = 0.7
    keep = pred["scores"] > score_threshold
    
    pred_boxes = pred["boxes"][keep].long()
    pred_scores = pred["scores"][keep]
    pred_labels = [f"pedestrian: {score:.3f}" for score in pred_scores]
    pred_masks = pred["masks"][keep]

    if len(pred_boxes) == 0:
        print("No objects detected with confidence > 0.7")
        return

    print(f"Found {len(pred_boxes)} pedestrians.")

    # Draw boxes
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red", width=2)

    # Draw masks
    masks = (pred_masks > 0.5).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    # Save result
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis('off')
    output_filename = "inference_result.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    main()