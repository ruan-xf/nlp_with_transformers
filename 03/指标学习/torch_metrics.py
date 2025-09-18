import torch
from data import y_true, y_pred

def torch_confusion_matrix(y_true, y_pred):
    y_true_t = torch.tensor(y_true)
    y_pred_t = torch.tensor(y_pred)
    classes = y_true_t.size(1)
    cm = torch.zeros(classes, 2, 2, dtype=torch.int32)
    
    for i in range(classes):
        tp = torch.sum((y_true_t[:,i] == 1) & (y_pred_t[:,i] == 1)).item()
        fp = torch.sum((y_true_t[:,i] == 0) & (y_pred_t[:,i] == 1)).item()
        fn = torch.sum((y_true_t[:,i] == 1) & (y_pred_t[:,i] == 0)).item()
        tn = torch.sum((y_true_t[:,i] == 0) & (y_pred_t[:,i] == 0)).item()
        cm[i] = torch.tensor([[tn, fp], [fn, tp]])
    
    return cm 