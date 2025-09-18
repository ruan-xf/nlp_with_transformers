import os
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 固定随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据加载与预处理
def load_and_preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train['content'] = train['content'].fillna('空值')
    test['content'] = test['content'].fillna('空值')
    for df in [train, test]:
        df['level_1'] = df['level_1'].apply(lambda x: x.split('（')[0])
        df['level_2'] = df['level_2'].apply(lambda x: x.split('）')[-1])
        df['level_3'] = df['level_3'].apply(lambda x: re.split(r'[0-9]、', x)[-1])
        df['level_4'] = df['level_4'].apply(lambda x: re.split(r'[0-9]、', x)[-1])
        df['text'] = df['content'] + '[SEP]' + df['level_1'] + '[SEP]' + df['level_2'] + '[SEP]' + df['level_3'] + '[SEP]' + df['level_4']
    return train, test

# 数据集定义
class EnterpriseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EnterpriseDataset(
        texts=df['text'].values,
        labels=df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)

# 模型定义
from transformers import BertModel
class EnterpriseDangerClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model_name):
        super(EnterpriseDangerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# 训练与验证

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

# 主流程
if __name__ == "__main__":
    # 路径根据实际情况修改
    train_path = 'train.csv'
    test_path = 'test.csv'
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    MAX_LEN = 160
    BATCH_SIZE = 4
    EPOCHS = 10
    class_names = [0, 1]

    train, test = load_and_preprocess_data(train_path, test_path)
    df_train, df_test = train_test_split(train, test_size=0.1, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = EnterpriseDangerClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
        print(f"Epoch {epoch+1}/{EPOCHS} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

    # 测试集评估
    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )
    print(f"Test accuracy: {test_acc:.4f}")

    # 推理示例
    def predict(text):
        encoded_text = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        return class_names[prediction]

    # 示例
    sample_text = "电源插头应按规定正确接线"
    print(f'Sample text: {sample_text}')
    print(f'Danger label  : {predict(sample_text)}') 