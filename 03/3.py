import ast
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer


# df = pd.read_pickle('example.pkl')
df = pd.read_csv('example.csv')
df['label'] = df.label.apply(ast.literal_eval)
# inputs, masks, labels, token_types = (
#     torch.tensor(uv pip sync requirements.txt
#         df[col_key].tolist()
#     )
#     for col_key in ['input_ids', 'attention_mask', 'label', 'token_type_ids']
# )


tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
    # "bert-base-uncased",
    r'C:\Users\Acer\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
)

sub_df = df.iloc[:10]

# 经尝试，这个用法和原来旧版本使用pad_to_max_length是一致的
encoded = tokenizer.batch_encode_plus(
    sub_df.comment_text.tolist(),
    max_length=100,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
    # pad_to_max_length=True
)

assert list(encoded.keys()) == ['input_ids', 'token_type_ids', 'attention_mask']

inputs, token_types, masks = encoded.values()
labels = torch.tensor(sub_df.label)

data = TensorDataset(inputs, token_types, masks, labels)

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
    dataloader = DataLoader(dataset, batch_size, is_shuffle)
    return dataloader

        

# encoded_df = pd.DataFrame(dict(encoded))

# encoded_df.input_ids.apply(len).value_counts()