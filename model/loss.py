import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def yolo_loss(predictions, targets, num_classes=80):
    """
    YOLO loss function (simplified)
    """
    batch_size = predictions.size(0)
    grid_size = predictions.size(2)
    
    # Reshape predictions: [batch, anchors*(5+classes), grid, grid] -> [batch, grid, grid, anchors, 5+classes]
    predictions = predictions.view(batch_size, 3, 5 + num_classes, grid_size, grid_size)
    predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()
    
    # Extract components
    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_cls = predictions[..., 5:]
    
    # Convert targets to same format (this is simplified)
    # In practice, you'd need proper target preparation
    target_boxes = targets['boxes'] if isinstance(targets, dict) else targets[:, :4]
    target_labels = targets['labels'] if isinstance(targets, dict) else targets[:, 4]
    
    # Compute losses (simplified)
    box_loss = F.mse_loss(pred_boxes, target_boxes.unsqueeze(-2).expand_as(pred_boxes))
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, torch.ones_like(pred_conf))
    cls_loss = F.cross_entropy(pred_cls.view(-1, num_classes), target_labels.long().view(-1))
    
    total_loss = box_loss + conf_loss + cls_loss
    return total_loss


def detection_loss(predictions, targets):
    """
    Simple detection loss for single-shot detector
    """
    # Extract predictions
    if isinstance(predictions, dict):
        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        pred_labels = predictions['labels']
    else:
        # Assume predictions are [batch, 5+classes, H, W]
        pred_boxes = predictions[:, :4]  # x, y, w, h
        pred_objectness = predictions[:, 4]
        pred_classes = predictions[:, 5:]
    
    # Extract targets
    if isinstance(targets, list):
        # Handle list of target dictionaries (typical for detection)
        target_boxes = torch.stack([t['boxes'] for t in targets])
        target_labels = torch.stack([t['labels'] for t in targets])
    else:
        target_boxes = targets[:, :4]
        target_labels = targets[:, 4].long()
    
    # Compute losses
    box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness, torch.ones_like(pred_objectness)
    )
    class_loss = F.cross_entropy(pred_classes, target_labels)
    
    return box_loss + objectness_loss + class_loss


def faster_rcnn_loss(output, targets=None):
    """
    Faster R-CNN loss - handles both training (loss dict) and validation modes
    
    Args:
        output: Either a loss dict (training) or predictions list (eval)
        targets: Ground truth targets (only used if output is predictions)
    """
    if isinstance(output, dict) and 'loss_classifier' in output:
        # Training mode: output is a loss dictionary
        return sum(loss for loss in output.values())
    else:
        # Evaluation mode: output is predictions, need to compute loss manually
        # This is a simplified version - you may want to implement proper detection loss
        return torch.tensor(0.0, requires_grad=True)