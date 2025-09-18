import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    Trainer,
    TrainingArguments,
)

swag = load_dataset("swag", "regular")
# datasets = swag

sm_ds = DatasetDict({
    'train': swag['train'].select(range(100)),
    'test': swag['test'].select(range(100))
})


model_checkpoint = "bert-base-uncased" # 使用的预训练模型
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def process_one(row):
    sent1 = [row['sent1']]*4
    ends = [row[name] for name in ["ending0", "ending1", "ending2", "ending3"]]
    sent2 = [f'{row['sent2']} {end}' for end in ends]
    return tokenizer(sent1, sent2, truncation=True)

# sm_ds = sm_ds.map(process_one)


tokenized_swag = swag.map(process_one)

args = TrainingArguments(
    f"{model_checkpoint}-finetuned-swag",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    # push_to_hub=True,
)



accuracy = evaluate.load("accuracy")





trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)