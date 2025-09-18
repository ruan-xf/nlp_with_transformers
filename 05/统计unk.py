import csv
import pandas as pd

data = pd.read_csv(
    'data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['char', 'tag'],
    # nrows=200,
    # skip_blank_lines=True
    skip_blank_lines=False,
    quoting=csv.QUOTE_NONE
)
# 先处理空行
null_rows = data.isnull().all(axis=1)
data['sentence_id'] = null_rows.cumsum()
data = data[~null_rows]

data.loc[pd.isna(data.tag), ['char', 'tag']] = ['[SEP]', 'O']

from transformers import AutoTokenizer, BertTokenizer

model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)

uniques = data.char.drop_duplicates()
tokens = uniques.apply(tokenizer.tokenize)
tokens.index = uniques

data['token'] = tokens[data.char].values


# 着重针对sentence_id做统计并记录，供后续定位
result = data[
    data.token.apply(lambda x: x==['[UNK]'])
].apply({
    'sentence_id': ['count', 'nunique', 'unique'],
    'char': 'unique'
})

result.to_csv('about_unk.csv')

