import csv
import pandas as pd

chars = pd.read_csv(
    'data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['token', 'tag'],
    # nrows=100,
    skip_blank_lines=True,
    # skip_blank_lines=False,
    quoting=csv.QUOTE_NONE,
).dropna()['token']

import time
from transformers import AutoTokenizer, BertTokenizer

model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} 耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

time_decorator(lambda: chars.apply(tokenizer.tokenize))()

def after_distinct():
    uniques = chars.drop_duplicates()
    tokens = uniques.apply(tokenizer.tokenize)
    results = pd.Series(tokens, index=uniques)[chars]
    print(results.shape)
    
time_decorator(after_distinct)()


# <lambda> 耗时: 115.3457 秒
# (2161197,)
# after_distinct 耗时: 0.6112 秒