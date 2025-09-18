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
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def get_device():
    if torch.cuda.is_available():
        logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
        logging.info(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logging.info("No GPU available, using the CPU instead.")
        return torch.device("cpu")

def load_and_preprocess_data(file_path):
    logging.info(f"Loading and preprocessing data from {file_path}")
    df = pd.read_csv(file_path)
    
    label_cols = list(df.columns[2:])
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    df['one_hot_labels'] = list(df[label_cols].values)
    
    labels = list(df.one_hot_labels.values)
    comments = list(df.comment_text.values)
    return df, comments, labels, label_cols

def tokenize_and_create_dataloaders(df, comments, labels, tokenizer, max_length, batch_size):
    logging.info("Tokenizing comments...")
    encodings = tokenizer.batch_encode_plus(
        comments,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    input_ids = encodings['input_ids']
    token_type_ids = encodings['token_type_ids']
    attention_masks = encodings['attention_mask']

    # Identify and handle labels with only one instance
    label_counts = df.one_hot_labels.astype(str).value_counts()
    one_freq = label_counts[label_counts==1].keys()
    one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
    
    one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
    one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
    one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
    one_freq_labels = [labels.pop(i) for i in one_freq_idxs]

    # Split data
    (train_inputs, validation_inputs, train_labels, validation_labels, 
     train_token_types, validation_token_types, train_masks, validation_masks) = train_test_split(
        input_ids, labels, token_type_ids, attention_masks,
        random_state=2020, test_size=0.10, stratify = labels
    )

    # Add one frequency data to train data
    train_inputs.extend(one_freq_input_ids)
    train_labels.extend(one_freq_labels)
    train_masks.extend(one_freq_attention_masks)
    train_token_types.extend(one_freq_token_types)

    # Convert to tensors
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_token_types = torch.tensor(validation_token_types)
    
    # Create dataloaders
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    logging.info("Dataloaders created.")
    return train_dataloader, validation_dataloader

def train_model(model, train_dataloader, validation_dataloader, optimizer, device, epochs, num_labels, save_path):
    model.to(device)
    best_val_f1 = 0

    for epoch in trange(epochs, desc="Epoch"):
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


def evaluate(model, dataloader, device, num_labels, threshold=0.5):
    model.eval()
    logit_preds, true_labels, pred_labels = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

        b_logit_pred = b_logit_pred.detach().cpu().numpy()
        pred_label = pred_label.to('cpu').numpy()
        b_labels = b_labels.to('cpu').numpy()

        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    
    val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100
    
    return val_f1_accuracy, val_flat_accuracy


def main(args):
    setup_logging()
    device = get_device()
    
    df, comments, labels, label_cols = load_and_preprocess_data(args.train_file)
    num_labels = len(label_cols)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    
    train_dataloader, validation_dataloader = tokenize_and_create_dataloaders(
        df, comments, labels, tokenizer, args.max_length, args.batch_size
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    train_model(model, train_dataloader, validation_dataloader, optimizer, device, args.epochs, num_labels, args.model_save_path)
    
    logging.info("Training complete.")
    logging.info(f"Model saved to {args.model_save_path}")

    # The following section for test set evaluation is commented out
    # because test files are not available in the current workspace.
    # You can uncomment it and provide the necessary files if you wish to run it.
    """
    if os.path.exists(args.test_file) and os.path.exists(args.test_labels_file):
        logging.info("Loading and evaluating on test data.")
        test_df = pd.read_csv(args.test_file)
        test_labels_df = pd.read_csv(args.test_labels_file)
        test_df = test_df.merge(test_labels_df, on='id', how='left')
        test_df = test_df[~test_df[label_cols].eq(-1).any(axis=1)]

        test_comments = list(test_df.comment_text.values)
        test_labels = list(test_df[label_cols].values)

        test_encodings = tokenizer.batch_encode_plus(
            test_comments, max_length=args.max_length, padding='max_length', truncation=True
        )

        test_inputs = torch.tensor(test_encodings['input_ids'])
        test_labels_tensor = torch.tensor(test_labels)
        test_masks = torch.tensor(test_encodings['attention_mask'])
        test_token_types = torch.tensor(test_encodings['token_type_ids'])

        test_data = TensorDataset(test_inputs, test_masks, test_labels_tensor, test_token_types)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)

        test_f1, test_acc = evaluate(model, test_dataloader, device, num_labels)
        logging.info(f"Test F1: {test_f1}, Test Accuracy: {test_acc}")

    else:
        logging.warning("Test files not found. Skipping test set evaluation.")
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for multi-label text classification.")
    parser.add_argument("--train_file", type=str, default="train.csv", help="Path to the training data CSV file.")
    # parser.add_argument("--test_file", type=str, default="test.csv", help="Path to the test data CSV file.")
    # parser.add_argument("--test_labels_file", type=str, default="test_labels.csv", help="Path to the test labels CSV file.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the pretrained model to use.")
    parser.add_argument("--model_save_path", type=str, default="bert_model_toxic.pt", help="Path to save the trained model.")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    main(args)
