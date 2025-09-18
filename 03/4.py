from transformers import AutoTokenizer, BertTokenizer

tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
    # "bert-base-uncased",
    r'C:\Users\Acer\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
)