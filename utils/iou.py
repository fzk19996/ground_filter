import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def load_label(file_name):
    label = np.fromfile(file_name, dtype=np.uint32)
    label = label.reshape((-1))
    return label

def calc_iou(pred, label_path):
    labels = load_label(label_path)
    invalid_idx = np.where(labels == 0)[0]
    labels_valid = np.delete(labels, invalid_idx)
    pred_valid = np.delete(pred, invalid_idx)
    correct = np.sum(pred_valid == labels_valid)
    true_postive = correct
    gt = np.where(labels==40)[0].shape[0]
    pred_positive = np.where(pred==40)[0].shape[0]
    return float(true_postive/(gt+pred_positive-true_postive))
