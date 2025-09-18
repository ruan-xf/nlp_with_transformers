import pandas as pd
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel, BertModel,
    get_linear_schedule_with_warmup
)
from torchmetrics.functional.classification import accuracy
import pytorch_lightning.loggers as pl_loggers


model_name = 'bert-base-chinese'
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
max_len = 160
max_epochs = 10
num_classes = 2
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集定义
class EnterpriseDataset(Dataset):
    def __init__(self, df_type: str):
        self.df = pd.read_csv(f'mydata/{df_type}.csv')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        text, label = self.df.iloc[item][['text', 'label']]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# dataset = EnterpriseDataset('val')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
# batch = next(iter(dataloader))

class EnterpriseDangerClassifier(pl.LightningModule):
    def __init__(self):
        super(EnterpriseDangerClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)


    def training_step(self, batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, targets) 

        acc = accuracy(outputs, targets, task='multiclass', num_classes=num_classes)
        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, targets) 

        acc = accuracy(outputs, targets, task='multiclass', num_classes=num_classes)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        

    def dataloader(self, train: bool):
        dataset = EnterpriseDataset('train' if train else 'val')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train)
        return dataloader

    def train_dataloader(self):
        return self.dataloader(True)

    def val_dataloader(self):
        return self.dataloader(False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        # total_steps = len(self.train_dataloader()) * max_epochs
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    model = EnterpriseDangerClassifier()
    tensorboard = pl_loggers.TensorBoardLogger('tb_logs', None)
    trainer = pl.Trainer(
        logger=tensorboard,
        max_epochs = max_epochs
    )
    trainer.fit(model)
