from data import y_true, y_pred
from numpy_metrics import np_confusion_matrix
from sklearn_metrics import sk_confusion_matrix
from torch_metrics import torch_confusion_matrix

def main():
    print("--------- NumPy Results ---------")
    np_cm = np_confusion_matrix(y_true, y_pred)
    print("NumPy Confusion Matrices:\n", np_cm)

    print("\n--------- Scikit-learn Results ---------")
    sk_cm = sk_confusion_matrix(y_true, y_pred)
    print("Scikit-learn Confusion Matrices:\n", sk_cm)

    print("\n--------- PyTorch Results ---------")
    torch_cm = torch_confusion_matrix(y_true, y_pred)
    print("PyTorch Confusion Matrices:\n", torch_cm)
    
    print("\n\n### 结果解释")
    print("所有三种方法将输出6个混淆矩阵（对应A-F六个类别），每个矩阵形式为：")
    print("""
    [[TN FP]
     [FN TP]]
    """)
    print("例如，对于类别A：")
    print("- 真正例(TP)：预测为A且实际为A的次数")
    print("- 假正例(FP)：预测为A但实际不是A的次数")
    print("- 假反例(FN)：实际为A但预测不是A的次数")
    print("- 真反例(TN)：实际不是A且预测也不是A的次数")

if __name__ == "__main__":
    main() 