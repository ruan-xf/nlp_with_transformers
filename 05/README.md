
数据格式：训练集数据每一行
- 第一列为一个字符或空格（汉字、英文字母、数字、标点符号、特殊符号、空格）
- 第二列为BIO形式的标签，两列以空格分隔。

实体共有52种类型，均已经过脱敏处理，用数字代号1至54表示（不包含27和45）；
- 其中“O”为非实体
- 标签中“B”代表一个实体的开始
- “I”代表一个实体的中间或者结尾
- “-”后的数字代号表示该字符的实体类型。
两条标注样本之间以空行为分割。

类似于csv，但第一列中的空格或引号又会影响解析
引号处理
`quoting=csv.QUOTE_NONE,`

空格空行的处理：
由于可利用空行给定样本id，读取时就不能跳过了，会保留为整行的na
```
sep=r'\s+',
skip_blank_lines=False,
```


token是单字符，显然可以按样本连接成句后查看
对空行的处理：
- 找出整行为na的，得到一个布尔类型的 Series 
- 给定样本id，利用.cumsum() 会逐行计算 True 的累积数量的特点
- 去掉空行
```py
null_rows = data.isnull().all(axis=1)
data['sentence_id'] = null_rows.cumsum()
data = data[~null_rows]
```

然后，列分隔符是空格同时token也是空格的情况下，前面设置的解析会让tag列为na
```py
# 未设置样本id时：
data[pd.isna(data.tag)] = [None, 'O']

# 设置样本id后：
data.loc[pd.isna(data.tag), ['token', 'tag']] = [None, 'O']
```

好，可以利用id连接成句了
```py
groups = data.groupby('sentence_id')

groups['token'].agg(lambda x: ''.join(x.fillna('\n'))).tolist()
groups.apply(lambda x: ''.join(x.token.fillna('\n')), include_groups=False).tolist()
```

apply可以接收单组的df，其中可以进行多列操作，相当灵活
当然，分组列也可以访问到（如果没有显式指定使用的列或include_groups=False），最好不要动就是了，现在也有警告提醒

相关说明，得空再研究区别
```
Series.groupby.apply : Apply function func group-wise and combine the results together.

Series.groupby.transform : Transforms the Series on each group based on the given function.

Series.aggregate : Aggregate using one or more operations over the specified axis.
```


```
74	定	B-29
75	制	I-29
76	l	I-29
77	o	I-29
78	g	I-29
79	o	I-29
```

如果直接连接成句：
tokenizer.decode(tokenizer.encode('定制logo'))

将切分为：'定 制 logo'
此时词和字符级标注的不对齐将显著影响模型精度

需要以空格连接
```py
# [CLS] 定 制 l o g o [SEP]
tokenizer.decode(tokenizer.encode('定 制 l o g o'))
```

原文档中用单词说明BertTokenizer的分词特点不太恰当，因为数据集是字符级标注，我不清楚字符分词还会不会被分几个部分


我也试了一下：
```py
chars = pd.read_csv(
    'data/train.txt',
    sep=r'\s+',
    header=None,
    names = ['token', 'tag'],
    skip_blank_lines=True, 
    quoting=csv.QUOTE_NONE,
).dropna()['token']

...

tokenized = chars.apply(tokenizer.tokenize)
tokenized.apply(len).value_counts()
```

没发现此现象，据~~ai~~说是有可能的，嗯

顺便看看字词重复的情况，2161197 -> 3533，恐怖如斯
那么对去重字符集进行分词再映射结果就应该是有点用的


可以使用encode_plus得到token_id啦！
- [快速 tokenizer 的特殊能力 - Hugging Face LLM Course](https://huggingface.co/learn/llm-course/zh-CN/chapter6/3)

感谢word_ids() 方法，可以获取每个 token 原始单词的索引

```py
result = tokenizer.encode_plus('doing that make me better')

result.word_ids() 
# [None, 0, 0, 1, 2, 3, 4, 4, None]

result.tokens()
# ['[CLS]', 'doi', '##ng', 'that', 'make', 'me', 'be', '##tter', '[SEP]']

```

这样只要：
- 有原始单词索引的，获取其标签，划分多部分的对应同一个标签
- 其他的设置无实体标签
就完成了原有的功能，而且可以预见到的，会更快




LightningDataModule会处理分布式训练，随之的是我需要充分理解数据加载的阶段：
- prepare_data()：全局一次性准备数据
- setup()：进程相关的数据分配

之前是直接在setup中初始化自己封装的torch Dataset
数据需要编码为token id，所以需要一个tokenizer并选择了在该dataset init中初始化
但这次有考虑到：
- tokenizer重复初始化的问题
- 同时对大量文本进行处理时，Transformers tokenizer会有优化

所以这次决定把原先torch Dataset的逻辑拆了放在dataModule中，处理完所有数据后再在setup中设置各阶段数据（Subset


```
(method) def prepare_data(self: Self@MyDefaultDataModule) -> None
Use this to download and prepare data. Downloading and saving data with multiple processes (distributed settings) will result in corrupted data. Lightning ensures this method is called only within a single process, so you can safely add your downloading logic within.

warning

DO NOT set state to the model (use setup instead) since this is NOT called on every device

Example:

    def prepare_data(self):
        # good
        download_data()
        tokenize()
        etc()

        # bad
        self.split = data_split
        self.some_state = some_other_state()

```

所以prepare_data()中不应该设置实例状态


Tokenizer的初始化时机？
- 