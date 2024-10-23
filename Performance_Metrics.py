import torch
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # Thresholding for binary masks
    y_pred = (y_pred > 0.5).float()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def intersection_over_union(y_true, y_pred, smooth=1e-6):
    # Thresholding for binary masks
    y_pred = (y_pred > 0.5).float()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()  # Thresholding
    correct = (y_true == y_pred).float().sum()  # Count of correct predictions
    return correct / y_true.numel()  # Total number of pixels

def sensitivity(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()  # Thresholding
    true_positives = (y_true * y_pred).sum()
    possible_positives = y_true.sum()
    return true_positives / (possible_positives + 1e-6)  # Avoid division by zero

def specificity(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()  # Thresholding
    true_negatives = ((1 - y_true) * (1 - y_pred)).sum()
    possible_negatives = (1 - y_true).sum()
    return true_negatives / (possible_negatives + 1e-6)  # Avoid division by zero
