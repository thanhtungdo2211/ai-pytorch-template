import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def detection_accuracy(output, target, iou_threshold=0.5):
    """
    Calculate detection accuracy based on IoU threshold
    
    Args:
        output: Model predictions (list of dicts with 'boxes', 'labels', 'scores')
        target: Ground truth (list of dicts with 'boxes', 'labels')
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        accuracy: Detection accuracy
    """
    if not isinstance(output, list):
        return 0.0
    
    total_correct = 0
    total_gt = 0
    
    for pred, gt in zip(output, target):
        if 'boxes' not in pred or 'boxes' not in gt:
            continue
            
        pred_boxes = pred['boxes'].cpu().detach()
        gt_boxes = gt['boxes'].cpu().detach()
        
        if len(gt_boxes) == 0:
            continue
            
        total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        # Calculate IoU between all pred and gt boxes
        ious = box_iou(pred_boxes, gt_boxes)
        
        # For each gt box, check if any prediction matches
        max_ious, _ = ious.max(dim=0)
        total_correct += (max_ious >= iou_threshold).sum().item()
    
    if total_gt == 0:
        return 0.0
    
    return total_correct / total_gt


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) tensor in format [x1, y1, x2, y2]
        boxes2: (M, 4) tensor in format [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) tensor of IoU values
    """
    import torch
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expand dimensions for broadcasting
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def mean_average_precision(output, target, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        output: Model predictions (list of dicts with 'boxes', 'labels', 'scores')
        target: Ground truth (list of dicts with 'boxes', 'labels')
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        mAP: mean Average Precision
    """
    if not isinstance(output, list):
        return 0.0
    
    # Collect all predictions and ground truths per class
    all_predictions = {}
    all_ground_truths = {}
    
    for pred, gt in zip(output, target):
        if 'boxes' not in pred or 'boxes' not in gt:
            continue
        
        # Ground truths
        gt_boxes = gt['boxes'].cpu().detach()
        gt_labels = gt['labels'].cpu().detach()
        
        for label in gt_labels.unique():
            label = label.item()
            if label not in all_ground_truths:
                all_ground_truths[label] = []
            
            mask = gt_labels == label
            all_ground_truths[label].append(gt_boxes[mask])
        
        # Predictions
        if 'labels' in pred and 'scores' in pred:
            pred_boxes = pred['boxes'].cpu().detach()
            pred_labels = pred['labels'].cpu().detach()
            pred_scores = pred['scores'].cpu().detach()
            
            for label in pred_labels.unique():
                label = label.item()
                if label not in all_predictions:
                    all_predictions[label] = []
                
                mask = pred_labels == label
                all_predictions[label].append({
                    'boxes': pred_boxes[mask],
                    'scores': pred_scores[mask]
                })
    
    if len(all_ground_truths) == 0:
        return 0.0
    
    # Calculate AP for each class
    aps = []
    for label in all_ground_truths.keys():
        if label not in all_predictions:
            aps.append(0.0)
            continue
        
        # Simplified AP calculation (you can implement a more sophisticated version)
        ap = 0.5  # Placeholder - implement proper AP calculation
        aps.append(ap)
    
    return sum(aps) / len(aps) if len(aps) > 0 else 0.0