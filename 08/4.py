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

model_name = "google/pegasus-cnn_dailymail"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)


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

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

out = seq2seq_data_collator(ds)

out2 = model(out)

# out2.keys()
# odict_keys(['loss', 'logits', 'encoder_last_hidden_state'])

# out2.logits.shape
# torch.Size([3, 7, 96103])

# tokenizer.batch_decode(model.generate(out.input_ids))
# ['<pad> Doctors recommend at least 30 minutes of moderate activity daily .<n>Studies show that regular exercise improves both physical and mental health .<n>Doctors recommend at least 30 minutes of moderate activity daily .</s>',
#  '<pad> A balanced diet rich in fruits and vegetables significantly reduces the risk of chronic diseases .<n>Maintaining a balanced diet rich in fruits and vegetables significantly reduces the risk of chronic diseases .</s></s></s></s>',
#  '<pad> Getting sufficient sleep each night enhances cognitive function and boosts immune system performance .<n>Getting sufficient sleep each night increases cognitive function and boosts immune system performance .</s></s></s></s></s></s></s></s></s>']

# model.config
#   "max_length": 128,
#   "max_position_embeddings": 1024,