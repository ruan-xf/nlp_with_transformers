from sklearn.metrics import multilabel_confusion_matrix
from data import y_true, y_pred

# In multilabel confusion matrix MCM,
# the count of true negatives is MCM_{:,0,0},
# false negatives is MCM_{:,1,0},
# true positives is MCM_{:,1,1}
# and false positives is MCM_{:,0,1}.
# 可按索引访问，即[[tn, fp], [fn, tp]]
def sk_confusion_matrix(y_true, y_pred):
    return multilabel_confusion_matrix(y_true, y_pred) 

cm = sk_confusion_matrix(y_true, y_pred)

cm