from datasets import load_dataset

swag = load_dataset("swag", "regular")

small_swag = swag['train'].shuffle(1111).select(range(500)).train_test_split(.3)
small_swag['validation'] = swag['validation'].shuffle(1111).select(range(150))
# small_swag = DatasetDict()


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


ending_names = ["ending0", "ending1", "ending2", "ending3"]

def process_one(row):
    sent1 = [row['sent1']]*4
    ends = [row[name] for name in ["ending0", "ending1", "ending2", "ending3"]]
    sent2 = [f'{row['sent2']} {end}' for end in ends]
    return tokenizer(sent1, sent2, truncation=True)

# sm_ds = sm_ds.map(process_one)



tokenized_swag = small_swag.map(process_one)



from transformers import DataCollatorForMultipleChoice
collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)


import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")



training_args = TrainingArguments(
    output_dir="my_awesome_swag_model",
    eval_strategy="epoch",
    # save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["validation"],
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()



