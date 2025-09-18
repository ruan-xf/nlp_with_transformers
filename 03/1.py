import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                            TensorDataset)
from tqdm import tqdm, trange


from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('train.csv')
label_names = df.columns[2:]
num_labels = len(label_names[2:])
df['one_hot_labels'] = df.iloc[:, 2:].values.tolist()

# 将保持类别分布划分，但数目小于2的将出错，所以需要手动分离，之后再合到训练集
one_hot_labels = df.one_hot_labels.astype(str)
label_count = (
    one_hot_labels.groupby(one_hot_labels)
    .agg(count='count', row_ids=lambda d: d.index)
)


one_freq_ids = label_count[label_count['count'] == 1].row_ids.astype(int)
one_freq_mask = np.zeros(df.shape[0], dtype=bool)
one_freq_mask[one_freq_ids] = 1
train_ids, valid_ids = train_test_split(
    np.where(~one_freq_mask)[0].tolist(),
    test_size=0.10,
    stratify=one_hot_labels[~one_freq_mask]
)
train_ids.extend(one_freq_ids)

model_name = "bert-base-uncased"
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def get_data(ids, is_shuffle=False):
    sub_df = df.iloc[ids]

    encoded = tokenizer.batch_encode_plus(
        sub_df.comment_text.tolist(),
        max_length=100,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    assert list(encoded.keys()) == ['input_ids', 'token_type_ids', 'attention_mask']

    inputs, token_types, masks = encoded.values()
    labels = torch.tensor(sub_df.label)

    dataset = TensorDataset(inputs, token_types, masks, labels)
    dataloader = DataLoader(dataset, 32, is_shuffle)
    return dataloader


train_dataloader, validation_dataloader = \
    get_data(train_ids, is_shuffle=True), get_data(valid_ids)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    logit_preds, true_labels, pred_labels = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

        b_logit_pred = b_logit_pred.tolist()
        pred_label = pred_label.tolist()
        b_labels = b_labels.tolist()

        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # pred_labels = [item for sublist in pred_labels for item in sublist]
    # true_labels = [item for sublist in true_labels for item in sublist]
    
    # pred_bools = [pl > threshold for pl in pred_labels]
    # true_bools = [tl == 1 for tl in true_labels]
    
    # val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
    # val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100
    val_f1_accuracy = f1_score(true_labels, pred_labels, average='micro') * 100
    val_flat_accuracy = accuracy_score(np.array(true_labels).flatten(), np.array(pred_labels).flatten()) * 100
    
    return val_f1_accuracy, val_flat_accuracy


model.to(device)

for epoch in trange(5, desc="Epoch"):
    model.train()
    tr_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        
        optimizer.zero_grad()
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        
        loss_func = BCEWithLogitsLoss()
        loss = loss_func(logits.view(-1, num_labels), b_labels.type_as(logits).view(-1, num_labels))
        
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item()

    avg_train_loss = tr_loss / len(train_dataloader)
    logging.info(f"Train loss: {avg_train_loss}")

    val_f1, val_accuracy = evaluate(model, validation_dataloader, device, num_labels)
    logging.info(f"Validation F1: {val_f1}, Validation Accuracy: {val_accuracy}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        logging.info(f"New best F1 score: {best_val_f1}. Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)