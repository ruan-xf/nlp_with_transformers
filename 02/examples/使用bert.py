import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModel, BertModel

content = "使用移动手动电动工具,外接线绝缘皮破损,应停止使用."
levels = ('工业/危化品类',
 '电气安全',
 '移动用电产品、电动工具及照明',
 '移动使用的用电产品和I类电动工具的绝缘线，必须采用三芯(单相)或四芯(三相)多股铜芯橡套软线。')

text = '[SEP]'.join([content, *levels])

model_name = 'bert-base-chinese'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

d = tokenizer.encode_plus(
    text,
    return_tensors='pt'
)
d

# _, pooled_output =
model(
    input_ids=d.input_ids,
    attention_mask=d.attention_mask,
    return_dict=False
)