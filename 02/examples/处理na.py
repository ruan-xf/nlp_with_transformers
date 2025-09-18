from io import StringIO
import numpy as np
import pandas as pd


raw = '''
A,B,C,D
,2,,0
3,4,,1
,,,
,3,,4
'''

df = pd.read_csv(StringIO(raw), dtype=str)

# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
# df.C.fillna('空值', inplace=True)
df.fillna({'C': '空'}, inplace=True)

x = df.A[0]
# (False, True)
x is pd.NA, pd.isna(x)


df1 = pd.read_csv('../data/train.csv')
x1 = df1.content[pd.isna(df1.content)].iloc[0]

# x: nan, x1: nan
# (False, True, True)
x == x1, pd.isna(x), pd.isna(x1)