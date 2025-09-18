from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split


data_root = Path('mydata')

def row2line(row):
    content = row.content
    if pd.isna(content): content = '空值'
    lv1, lv2, lv3, lv4 = (row[f'level_{i}'] for i in range(1,4+1))
    lv1, lv2 = (
        x.split(sep)[i]
        for x, sep, i in 
        zip((lv1,lv2), '（）', (0, -1))
    )
    lv3, lv4 = (
        re.split(r'[0-9]、', x)[-1]
        for x in (lv3, lv4)
    )
    return '[SEP]'.join((
        content,
        lv1, lv2, lv3, lv4
    ))


def get_full():
    df = pd.read_csv(data_root.joinpath('raw.csv'))
    df['text'] = df.apply(row2line, axis=1)
    df = df[['text', 'label']]
    df.to_csv(data_root.joinpath('full.csv'), index=False)


# 抽取的子集还会有full的index
def get_sub():
    df = pd.read_csv(data_root.joinpath('full.csv'))
    df = df.groupby(df.label).sample(140).sample(frac=1)
    # 打乱标签后就不该有聚集的同标签了
    # df.label.reset_index(drop=True).plot()
    df.to_csv(data_root.joinpath('sub.csv'))


# 反复调整每组抽样数，最终抽140，比例为(252, 14, 14)
# get_sub()

def get_split():
    df = pd.read_csv(data_root.joinpath('sub.csv'), index_col=0)
    df_train, df_test = train_test_split(df, test_size=.1, stratify=df.label)
    df_val, df_test = train_test_split(df_test, test_size=.5, stratify=df_test.label)
    print(len(df_train), len(df_val), len(df_test))
    for sub_df, name in zip((df_train, df_val, df_test), ('train', 'val', 'test')):
        sub_df.to_csv(data_root.joinpath(f'{name}.csv'))


# get_split()