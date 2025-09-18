import csv
import pandas as pd


ser = pd.read_csv(
    'data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['token', 'tag'],
    # chunksize=400,
    # nrows=200,
    usecols=[1],
    # skip_blank_lines=True
    skip_blank_lines=False,
    quoting=csv.QUOTE_NONE
).squeeze().fillna('O')

def convert(x:str):
    c, *n = x.split('-')
    n = next(iter(n), None)
    return pd.Series((c,n))

df = ser.apply(convert)
df.drop_duplicates().groupby(0).size()

# 实体共有52种类型，均已经过脱敏处理，用数字代号1至54表示（不包含27和45）；其中“O”为非实体。
# 0
# B    52
# I    52
# O     1
# dtype: int64


# labels = [
#     f'{c}-{i}'
#     for c in 'BI' for i in range(1, 54+1) if i not in (27, 45)
# ]

# labels.append('O')

