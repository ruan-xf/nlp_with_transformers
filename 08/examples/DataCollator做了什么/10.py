import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline, AutoModel, PreTrainedModel, PreTrainedTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Trainer, DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset
import evaluate


# dataset = load_dataset("knkarthick/samsum")

# model_name = "google/pegasus-cnn_dailymail"
model_name = 'sshleifer/distilbart-cnn-12-6'
# model_name = "t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)




# trainer = Trainer(
#     model,
# )



ds = Dataset.from_list([
    {
    "source_text": "Studies show that regular exercise improves both physical and mental health. Doctors recommend at least 30 minutes of moderate activity daily.",
    "target_text": "Exercise benefits health."
    },
    {
    "source_text": "Research indicates that maintaining a balanced diet rich in fruits and vegetables significantly reduces the risk of chronic diseases.",
    "target_text": "Healthy eating prevents illness."
    },
    {
    "source_text": "Getting sufficient sleep each night enhances cognitive function and boosts immune system performance.",
    "target_text": "Adequate sleep improves wellbeing."
    }
])

def process_one(row):
    return tokenizer(row['source_text'], text_target=row['target_text'], truncation=True)

ds = ds.map(process_one).remove_columns(['source_text', 'target_text'])

# model : [PreTrainedModel], optional
# The model that is being trained. If set and has the prepare_decoder_input_ids_from_labels, use it to prepare the decoder_input_ids
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

out = seq2seq_data_collator(ds)

# out.data.keys()
# dict_keys(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])

# out.input_ids.shape [3, 25]
# out.labels.shape [3, 9]

# tokenizer.batch_decode(out.input_ids)
# ['<s>Studies show that regular exercise improves both physical and mental health. Doctors recommend at least 30 minutes of moderate activity daily.</s>',
#  '<s>Research indicates that maintaining a balanced diet rich in fruits and vegetables significantly reduces the risk of chronic diseases.</s><pad><pad><pad>',
#  '<s>Getting sufficient sleep each night enhances cognitive function and boosts immune system performance.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad>']

# tokenizer.batch_decode(out.labels)
# OverflowError: can't convert negative int to unsigned
# 查看out.labels发现末尾填充了-100

# [t.shape for t in (out.decoder_input_ids, out.labels)]
# [torch.Size([3, 9]), torch.Size([3, 9])]

# out.decoder_input_ids
# tensor([[    2,     0,  9089, 42118,  1795,   474,     4,     2,     1],
#         [    2,     0, 13716,   219,  4441, 17410,  5467,     4,     2],
#         [    2,     0,  9167,  8198,   877,  3581, 15296, 19715,     4]])


# out.labels
# tensor([[    0,  9089, 42118,  1795,   474,     4,     2,  -100,  -100],
#         [    0, 13716,   219,  4441, 17410,  5467,     4,     2,  -100],
#         [    0,  9167,  8198,   877,  3581, 15296, 19715,     4,     2]])

# tokenizer.special_tokens_map
# {'bos_token': '<s>',
#  'eos_token': '</s>',
#  'unk_token': '<unk>',
#  'sep_token': '</s>',
#  'pad_token': '<pad>',
#  'cls_token': '<s>',
#  'mask_token': '<mask>'}

# tokenizer.added_tokens_encoder
# {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 50264}



# - [python - What is the function of the `text_target` parameter in Huggingface's `AutoTokenizer`? - Stack Overflow](https://stackoverflow.com/questions/76130589/what-is-the-function-of-the-text-target-parameter-in-huggingfaces-autotokeni)

# def decode(encoding):
#     return tokenizer.decode(encoding.input_ids)

# print(decode(tokenizer(source_text, max_length=30, padding='max_length', truncation=True)))

# # with tokenizer.as_target_tokenizer():
# #     print(decode(tokenizer(target_text, max_length=8, padding='max_length', truncation=True)))

# print(decode(tokenizer(text_target=target_text, max_length=8, padding='max_length', truncation=True)))
