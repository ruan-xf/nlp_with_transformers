import pandas as pd
# from transformers import AutoTokenizer, BertTokenizer


# conversation = '''\
# Hi! / Hello!
# Hello! How are you?
# I'm good, thanks! And you?
# Pretty well, thank you!'''.splitlines()

# tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
#     "bert-base-uncased",
# )

# data = tokenizer.batch_encode_plus(
#     conversation,
#     max_length=100,
#     padding='max_length',
# )

# tokenized_df = pd.DataFrame(dict(data))

# valid_data = pd.DataFrame({
#     'id': valid_ids,
#     'label': df.one_hot_labels[valid_ids],
#     'comment_text': df.comment_text[valid_ids]
# })
# valid_data.to_pickle('example.pkl')


df = pd.read_pickle('example.pkl')
df.to_csv('example.csv', index=False)