import pandas as pd
import csv


data = pd.read_csv(
    'data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['token', 'tag'],
    nrows=200,
    # skip_blank_lines=True
    skip_blank_lines=False,
    quoting=csv.QUOTE_NONE
)
# 先处理空行
null_rows = data.isnull().all(axis=1)
data['sentence_id'] = null_rows.cumsum()
data = data[~null_rows]

data.loc[pd.isna(data.tag), ['token', 'tag']] = ['[SEP]', 'O']

groups = data.groupby('sentence_id')

# sentence, tags = groups.apply(
#     lambda x: (''.join(x.token.fillna('\n')), x.tag.tolist()),
#     include_groups=False 
# )[0]

# Series.groupby.apply : Apply function func group-wise and combine the results together.

# Series.groupby.transform : Transforms the Series on each group based on the given function.

# Series.aggregate : Aggregate using one or more operations over the specified axis.