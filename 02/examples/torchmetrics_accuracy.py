import torch
from torch import tensor
from torchmetrics.functional.classification import accuracy


# 1
# target = tensor([0, 1, 2, 3])
# preds = tensor([0, 2, 1, 3])
# accuracy(preds, target, task="multiclass", num_classes=4)
# tensor(0.5000)



# 2
# target = tensor([0, 1, 2])
# preds = tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])

# accuracy(preds, tensor([2,0,1]), task="multiclass", num_classes=3)

# 3
accuracy(torch.tensor([0.3, 0.4, 0.7, 0.9]), torch.tensor([1, 0, 1, 0]), task='binary')


# 4
out = tensor([[-0.7735, -0.3202],
        [ 0.1754, -0.6127],
        [-0.3139, -1.0355],
        [ 0.2484,  0.1951]])

# torch.max(out, dim=1).indices
# tensor([1, 0, 0, 0])

accuracy(out, torch.tensor([1, 1, 1, 0]), task='multiclass', num_classes=2)