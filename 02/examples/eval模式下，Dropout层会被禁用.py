import torch
import torch.nn as nn

model = nn.Dropout(p=0.5)  # 50%的神经元会被随机丢弃
input = torch.ones(10)


# 训练模式（默认）
model.train()
output_train = model(input)
print("Train mode output:", output_train)  # 部分值会被置零

# 评估模式
model.eval()
output_eval = model(input)
print("Eval mode output:", output_eval)   # 所有值保留（无Dropout）