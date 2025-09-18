
# 预测集需要另外加载新的文件
# data/sample_per_line_preliminary_A.txt

# setup
import csv
import pandas as pd
import numpy as np

import os
os.chdir(os.path.dirname(__file__))

data = pd.read_csv(
    '../../data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['char', 'tag'],
    nrows=200,
    # skip_blank_lines=True
    skip_blank_lines=False,
    quoting=csv.QUOTE_NONE
)
# 先处理空行
null_rows = data.isnull().all(axis=1)
data['sentence_id'] = null_rows.cumsum()
data = data[~null_rows]

data.loc[pd.isna(data.tag), ['char', 'tag']] = ['[SEP]', 'O']



# 定义所有实体标签：B-{type}和I-{type}，排除特定类型
entity_labels = [
    f'{prefix}-{entity_type}'
    for prefix in ['B', 'I'] 
    for entity_type in range(1, 54+1) 
    if entity_type not in (27, 45)  # 排除不需要的实体类型
]

# 添加非实体标签'O'
all_labels = entity_labels + ['O']

label_id_to_name = pd.Series(all_labels)
label_name_to_id = pd.Series(label_id_to_name.index, index=all_labels)


data['label_id'] = label_name_to_id[data.tag].values

# 设置radio，抽取多少句子


# 划分训练和验证集

grouped = data[['sentence_id', 'char', 'label_id']].groupby('sentence_id')


data = grouped.apply(
    lambda x: pd.Series({
        'sentence': ' '.join(x.char),
        'labeling': x.reset_index(drop=True)
    }), include_groups=False
)

from transformers import AutoTokenizer, BertTokenizer


model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)

max_len = 105

# 对句子进行tokenization编码
encoded_sentences = tokenizer.batch_encode_plus(
    data.sentence.tolist(),
    add_special_tokens=True,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_attention_mask=True,
    return_token_type_ids=False,
)

# 获取每个token对应的原始单词位置
token_to_char_mapping = pd.Series(np.arange(len(data))).apply(encoded_sentences.word_ids)

def assign_labels_to_tokens(original_labels, token_char_mapping):
    # 为特殊token添加默认标签
    original_labels.loc[-1] = (None, label_name_to_id['O'])
    
    # 创建token到标签的映射表
    token_labels = pd.DataFrame({
        'char_position': pd.Series(token_char_mapping).fillna(-1).astype(int),
        'label': None
    })
    
    # 将原始单词标签分配给对应的token
    token_labels['label'] = original_labels.loc[token_labels.char_position, 'label_id'].values
    
    return token_labels.label.tolist()

# 为每个句子的token分配标签
outputs = pd.concat([data.labeling, token_to_char_mapping], axis=1).apply(
    lambda row: assign_labels_to_tokens(*row),
    axis=1
).tolist()

import torch
outputs = torch.LongTensor(outputs)
encoded_sentences['outputs'] = outputs
encoded_sentences.data.keys()

from torch.utils.data import TensorDataset


assert list(encoded_sentences.data.keys()) == ['input_ids', 'attention_mask', 'outputs']
ds = TensorDataset(*encoded_sentences.values())