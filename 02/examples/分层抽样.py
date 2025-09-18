import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义类别标签
categories = ['A', 'B', 'C', 'D']

# 定义样本数量比例 6:3:3:1
total_samples = 130  # 选择130以便比例分配为整数
n_A = 60    # 6/13 ≈ 60/130
n_B = 30    # 3/13 ≈ 30/130
n_C = 30    # 3/13 ≈ 30/130
n_D = 10    # 1/13 ≈ 10/130

labels = np.concatenate([
    ['A'] * n_A,
    ['B'] * n_B,
    ['C'] * n_C,
    ['D'] * n_D
])

# 打乱数据顺序
indices = np.arange(total_samples)
np.random.shuffle(indices)
labels = labels[indices]

ser = pd.Series(labels)
# ser.value_counts(normalize=True)
# A    0.461538
# C    0.230769
# B    0.230769
# D    0.076923
# Name: proportion, dtype: float64


# 每次运行的比例都和原分布一致
train_idx, test_idx = train_test_split(ser.index, stratify=ser)
ser[test_idx].value_counts(normalize=True)

# 每次运行的比例都不一样，大概率都和原分布不一致的
train_idx, test_idx = train_test_split(ser.index)
ser[test_idx].value_counts(normalize=True)