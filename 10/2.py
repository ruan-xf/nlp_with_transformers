from transformers import BertTokenizer, AutoTokenizer

tokenizer: BertTokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"

encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])