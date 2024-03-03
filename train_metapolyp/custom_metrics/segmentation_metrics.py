import torch
import torch.nn.functional as F

def dice_coeff(y_true, y_pred):
    epsilon = 1e-7
    intersections = torch.sum(y_true * y_pred)
    unions = torch.sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + epsilon) / (unions + epsilon)
    
    return dice_scores

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    return 0.5 * F.binary_cross_entropy(y_pred, y_true) + dice_loss(y_true, y_pred)

def IoU(y_true, y_pred, eps=1e-6):
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3]) - intersection
    return torch.mean((intersection + eps) / (union + eps), dim=0)

def zero_IoU(y_true, y_pred):
    return IoU(1 - y_true, 1 - y_pred)

def bce_dice_loss(y_true, y_pred):
    return F.binary_cross_entropy(y_pred, y_true) + dice_loss(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return torch.pow((1 - tv), gamma)