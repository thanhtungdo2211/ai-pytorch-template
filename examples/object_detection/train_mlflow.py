import torch
import os
import sys
import mlflow
import mlflow.pytorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.object_detection.engine import train_one_epoch, evaluate
from examples.object_detection import utils

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def _log_coco_metrics(coco_evaluator, epoch):
    # COCOeval.stats order:
    # [AP, AP50, AP75, APs, APm, APl, AR@1, AR@10, AR@100, ARs, ARm, ARl]
    stat_names = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR@1",
        "AR@10",
        "AR@100",
        "ARs",
        "ARm",
        "ARl",
    ]
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        if coco_eval.stats is None or len(coco_eval.stats) == 0:
            continue
        metrics = {
            f"{iou_type}/{name}": float(value)
            for name, value in zip(stat_names, coco_eval.stats)
        }
        mlflow.log_metrics(metrics, step=epoch)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

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

# train on the accelerator or on the CPU, if an accelerator is not available
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, 
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 1

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "object_detection"))

with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME")):
    mlflow.log_params(
        {
            "num_classes": num_classes,
            "batch_size": data_loader.batch_size,
            "lr": optimizer.param_groups[0]["lr"],
            "momentum": optimizer.param_groups[0].get("momentum"),
            "weight_decay": optimizer.param_groups[0].get("weight_decay"),
            "step_size": lr_scheduler.step_size,
            "gamma": lr_scheduler.gamma,
            "num_epochs": num_epochs,
            "model": "maskrcnn_resnet50_fpn",
        }
    )

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)

        # log averaged training losses + lr
        train_metrics = {
            f"train/{name}": meter.global_avg
            for name, meter in metric_logger.meters.items()
        }
        mlflow.log_metrics(train_metrics, step=epoch)

        # log COCO evaluation metrics
        _log_coco_metrics(coco_evaluator, epoch)

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    mlflow.log_artifact("model.pth")
    print("Model saved to model.pth")

    print("That's it!")
