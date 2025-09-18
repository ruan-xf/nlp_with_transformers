import pytorch_lightning as pl
import torch
from torch import nn

class MinimalModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)  # 示例网络层
    
    # 必须实现的方法1：训练步骤
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)  # 自动记录日志
        return loss
    
    # 必须实现的方法2：优化器配置
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    # 必须实现的方法3：训练数据加载
    def train_dataloader(self):
        # 示例虚拟数据（实际使用时替换为真实DataLoader）
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),  # X
            torch.randn(100, 1)     # Y
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32)

model = MinimalModel()
# 创建虚拟Trainer进行验证
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",
    enable_checkpointing = False,
    # fast_dev_run=True,  # 只运行1个batch用于验证
    logger=False        # 禁用日志
)
trainer.fit(model)  # 现在不会报错了

model.trainer.estimated_stepping_batches