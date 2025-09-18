from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup
)
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional.classification import accuracy

from data import CustomDataset

class SentencePairClassifier(pl.LightningModule):
    def __init__(
        self,
        bert_model="albert-base-v2",
        freeze_bert=False,
        lr=2e-5,
        weight_decay=1e-2,
        num_warmup_steps = 0,
        maxlen=128,
        batch_size=16
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        hidden_size = self.bert_layer.config.hidden_size
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(p=0.1)
        self.cls_layer = nn.Linear(hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 设置示例输入（模拟一个 batch=batch_size 的输入）
        self.example_input_array = (
            torch.randint(0, 100, (batch_size, 128)),  # token_ids (模拟随机词ID)
            torch.ones(batch_size, 128, dtype=int),               # attn_masks (全1掩码)
            torch.zeros(batch_size, 128, dtype=int),              # token_type_ids (全0表示单段落)
            # 注意：不包含 label
        )

    def forward(self, input_ids, attn_masks, token_type_ids):
        outputs = self.bert_layer(input_ids, attn_masks, token_type_ids)
        logits = self.cls_layer(self.dropout(outputs['pooler_output']))
        return logits

    def _shared_step(self, batch):
        input_ids, attn_masks, token_type_ids, labels = batch
        logits = self(input_ids, attn_masks, token_type_ids).squeeze(-1)
        loss = self.criterion(logits, labels.float())
        acc = accuracy(logits, labels, 'binary')
        return loss, acc

    def _shared_eval_step(self, batch, stage):
        loss, acc = self._shared_step(batch)
        self.log(f'{stage}_loss', loss, on_epoch=True)
        self.log(f'{stage}_acc', acc, on_epoch=True, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, 'test')

    def dataloader(self, df_type: str, train: bool):
        ds = CustomDataset(
            df_type,
            self.hparams.maxlen,
            self.hparams.bert_model
        )
        return torch.utils.data.DataLoader(ds, self.hparams.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.dataloader('train', True)

    def test_dataloader(self):
        return self.dataloader('test', False)

    def val_dataloader(self):
        return self.dataloader('validation', False)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return [optimizer], [scheduler]
