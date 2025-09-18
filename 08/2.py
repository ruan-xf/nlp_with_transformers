import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline, AutoModel, PreTrainedModel, PreTrainedTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Trainer, DataCollatorForSeq2Seq,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
import evaluate


dataset = load_dataset("knkarthick/samsum")

model_name = "google/pegasus-cnn_dailymail"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)


def process_one(row):
    return tokenizer(row['source_text'], text_target=row['target_text'], truncation=True)

dataset = dataset.map(process_one).remove_columns(['source_text', 'target_text'])

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10, 
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16)



trainer = Trainer(
    model=model, args=training_args,
    tokenizer=tokenizer, data_collator=seq2seq_data_collator,
    train_dataset=dataset["train"], 
    eval_dataset=dataset["validation"]
)

