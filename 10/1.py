from transformers import AutoModel, AutoTokenizer, BertTokenizer



model_checkpoint = "bert-base-cased"
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from datasets import load_dataset

raw_datasets = load_dataset("squad")

inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# inputs = tokenizer(
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )