import pandas as pd
import numpy as np


with open(
    '../../data/sample_per_line_preliminary_A.txt',
) as f:
    sentences = pd.Series(
        line for i, line in enumerate(f) if i < 200
    )


sentences = sentences.apply(
    lambda x: (
        pd.Series(list(x))
        .str.replace(r'\s+', '[SEP]', regex=True)
        .tolist()
    )
).str.join(' ').tolist()

from transformers import AutoTokenizer, BertTokenizer


model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)


max_len = 105

encoded_sentences = tokenizer.batch_encode_plus(
    sentences,
    add_special_tokens=True,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_attention_mask=True,
    return_token_type_ids=False,
)