import csv
import pandas as pd
# from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, BertTokenizer

import torch
from torch.utils.data import Dataset



model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)


# tokenizer.tokenize('\n')
# # [] 空白符为空，保持原文档的处理，先设置为'[SEP]'

# # {'unk_token': '[UNK]',
# #  'sep_token': '[SEP]',
# #  'pad_token': '[PAD]',
# #  'cls_token': '[CLS]',
# #  'mask_token': '[MASK]'}
# tokenizer.special_tokens_map

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df_type: str, maxlen, bert_model):
        self.data = pd.read_csv(f'data/train.txt')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):