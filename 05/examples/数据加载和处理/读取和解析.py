import pandas as pd

# 不错，会自动处理行首尾空格
raw = '''
手 B-40
  机 I-40  
三 B-4 
  O  
'''

from io import StringIO

data = pd.read_csv(
    # 'data/train.txt',
    StringIO(raw),
    sep=r'\s+',
    header=None,
    names = ['token', 'tag'],
    nrows=100,
    skip_blank_lines=True
)

# # token是空格同时列分隔符也是空格的情况下，才会出现第二列为na
# data[pd.isna(data.tag)] = ['[SEP]', 'O']


data