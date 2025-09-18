from transformers import (
    BertForTokenClassification,
)
import torch
import pytorch_lightning as pl
from torchmetrics.functional.classification import accuracy

from data import MyDefaultDataModule

class NER_Model(pl.LightningModule):
    def __init__(
        self,
        bert_model='hfl/chinese-roberta-wwm-ext',
        num_labels = len(MyDefaultDataModule.all_labels),
        learning_rate=2e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForTokenClassification.from_pretrained(
            bert_model,
            num_labels=num_labels
        )
        
        # 设置示例输入
        # assert list(encoded_sentences.keys()) == ['input_ids', 'attention_mask', 'outputs']
        # [t.shape for t in batch]
        # [torch.Size([32, 105]), torch.Size([32, 105]), torch.Size([32, 105])]
        self.example_input_array = (
            torch.zeros((32, 105), dtype=torch.long), 
            torch.zeros((32, 105), dtype=torch.long), 
            torch.zeros((32, 105), dtype=torch.long)
        )


    def forward(self, input_ids, attn_masks, labels=None):
        return self.model(input_ids, attn_masks, labels=labels).values()

    def _shared_step(self, batch):
        labels = batch[-1]
        loss, logits = self(*batch)
        acc = accuracy(
            logits,
            labels,
            task='multiclass',
            num_classes=self.hparams.num_labels
        )
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

    def configure_optimizers(self):
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.learning_rate)
        return optim


# trainer = pl.Trainer(
#     fast_dev_run=True,
#     gradient_clip_val=5,
# )


# model = NER_Model()
# dm = MyDefaultDataModule()

# trainer.fit(model, datamodule=dm)
