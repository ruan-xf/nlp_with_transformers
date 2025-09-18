import numpy as np
from data import y_true, y_pred

def np_confusion_matrix(y_true, y_pred):
    classes = y_true.shape[1]
    cm = np.zeros((classes, 2, 2), dtype=int)
    
    for i in range(classes):
        tp = np.sum((y_true[:,i] == 1) & (y_pred[:,i] == 1))
        fp = np.sum((y_true[:,i] == 0) & (y_pred[:,i] == 1))
        fn = np.sum((y_true[:,i] == 1) & (y_pred[:,i] == 0))
        tn = np.sum((y_true[:,i] == 0) & (y_pred[:,i] == 0))
        cm[i] = [[tn, fp], [fn, tp]]
    
    return cm 


cm = np_confusion_matrix(y_true, y_pred)

'''
array([[[2, 0],
        [0, 3]],

       [[3, 0],
        [0, 2]],

       [[3, 0],
        [0, 2]],

       [[2, 1],
        [1, 1]],

       [[1, 1],
        [1, 2]],

       [[2, 1],
        [1, 1]]])
'''
comb = np.sum(cm, axis=0)