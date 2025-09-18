from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

swag = load_dataset("swag", "regular")
# datasets = swag

sm_ds = DatasetDict({
    'train': swag['train'].select(range(100)),
    'test': swag['test'].select(range(100))
})

model_checkpoint = "bert-base-uncased" # 使用的预训练模型
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

ending_names = ["ending0", "ending1", "ending2", "ending3"]


def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


# tokenized_swag = swag.map(preprocess_function, batched=True)

# examples = swag["train"][:5]
# preprocess_function(examples)


def process_one(row):
    sent1 = [row['sent1']]*4
    ends = [row[name] for name in ["ending0", "ending1", "ending2", "ending3"]]
    sent2 = [f'{row['sent2']} {end}' for end in ends]
    return tokenizer(sent1, sent2, truncation=True)

sm_ds.map(process_one)