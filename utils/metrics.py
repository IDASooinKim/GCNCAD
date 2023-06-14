import torch
import numpy as np


def node_classification_acc(
        prediction:np.ndarray, ground_truth:np.ndarray
)->np.ndarray:
    
    acc_map_pred = torch.zeros((8,))
    acc_map_gt = torch.zeros((8,))

    prediction = torch.argmax(prediction, dim=2)

    indices, counts=torch.unique(ground_truth, return_counts=True)
    indices = indices.type(torch.long)

    for i, idx in enumerate(indices):
        acc_map_gt[idx] = counts[i]


    indices, counts=torch.unique(prediction, return_counts=True)
    indices = indices.type(torch.long)

    for i, idx in enumerate(indices):
        acc_map_pred[idx] = counts[i]
    
    score = torch.div(acc_map_gt, acc_map_pred)

    score = torch.where(score==float("Inf"), 0, score)
    score = torch.nan_to_num(score, 0).detach().cpu().numpy()
    
    return score


def node_classification_IoU(
        prediction:np.ndarray, ground_truth:np.ndarray
)->np.ndarray:

    predicted_mask = torch.argmax(prediction, dim=1)[0].detach().cpu().numpy()
    ground_truth_mask = ground_truth[0].detach().cpu().numpy()

    iou_scores = []
    
    # Iterate over each class
    for class_id in range(8):
        # Create binary masks for the current class
        predicted_mask_class = (predicted_mask == class_id).astype(np.uint8)
        ground_truth_mask_class = (ground_truth_mask == class_id).astype(np.uint8)

        # Compute intersection and union masks
        intersection = np.logical_and(predicted_mask_class, ground_truth_mask_class)
        union = np.logical_or(predicted_mask_class, ground_truth_mask_class)

        # Calculate intersection and union areas
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)

        # Calculate IoU score for the current class
        if union_area == 0:
            iou_score = 0  # Handle the case where the union area is zero
        else:
            iou_score = (intersection_area / union_area) # * 100
        
        iou_scores.append(iou_score)


    return iou_scores
