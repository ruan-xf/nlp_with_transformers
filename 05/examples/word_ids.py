from transformers import AutoTokenizer, BertTokenizer


model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)

# result = tokenizer.encode_plus('doing that make me better')
# result.word_ids()

raw = '''
doing a
that b
make c
me d
better e
'''

import pandas as pd
import numpy as np

data = np.loadtxt(raw.splitlines(), dtype=str)
df = pd.DataFrame(data, columns=['word', 'label'])




sentence = ' '.join(df.word)
result = tokenizer.encode_plus(sentence)
df.loc[-1] = (None, 'O')
result_df = pd.DataFrame({
    'word_id': pd.Series(result.word_ids()).fillna(-1).astype(int),
    'label': None
})
result_df['label'] = df.loc[result_df.word_id, 'label'].values
result_df