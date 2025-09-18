import pytorch_lightning as pl
import torch.nn as nn
import torch

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, hidden_size=128, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()  # 保存所有传入的超参数
        
        # 使用超参数构建网络
        self.net = nn.Sequential(
            nn.Linear(10, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, 1)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    # 必须实现的方法3：训练数据加载
    def train_dataloader(self):
        # 示例虚拟数据（实际使用时替换为真实DataLoader）
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),  # X
            torch.randn(100, 1)     # Y
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32)
    
    def configure_optimizers(self):
        # 使用保存的学习率
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# 初始化时传入超参数
model = LitModel(learning_rate=0.01, hidden_size=256)


trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",
    enable_checkpointing = False,
    logger=False
)
trainer.fit(model) 

model.hparams