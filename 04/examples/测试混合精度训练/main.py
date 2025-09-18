import torch
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

scaler = GradScaler()  # 依赖 CUDA

def train():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 没有 GPU 时
    with autocast('cuda'):  # 需要 CUDA
        loss = model(torch.randn(1, 10)).sum()
    scaler.scale(loss).backward()  # 需要 CUDA
    scaler.step(optimizer)
    scaler.update()


train()