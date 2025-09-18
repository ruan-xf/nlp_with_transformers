
import ast
import pandas as pd
from io import StringIO

raw = '''
token	tokenized	label
doing	['doi', '##ng']	label-x
running	['run', '##ning']	label-y
'''

df = pd.read_csv(StringIO(raw), sep='\t')
df['tokenized'] = df.tokenized.apply(ast.literal_eval)


df_expanded = df.explode('tokenized')


# (
#     df_expanded
#     .drop('token', axis=1)
#     .rename({'tokenized': 'word', 'tag': 'label'}, axis=1)
#     .set_index('word').squeeze()
# )