import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline, AutoModel, PreTrainedModel,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from datasets import load_dataset
import evaluate

# Samsung/samsum 404, switch to knkarthick/samsum
# dataset_samsum = load_dataset("knkarthick/samsum")


# cnn_dailymail 'article', 'highlights'
# knkarthick/samsum 'dialogue', 'summary'
dataset = load_dataset("cnn_dailymail", "3.0.0")
test_data = dataset['test']
x, y = (test_data[key][:2] for key in ('article', 'highlights'))

# dataset = load_dataset("knkarthick/samsum")
# test_data = dataset['test']
# x, y = (test_data[key][:2] for key in ('dialogue', 'summary'))


rouge_metric = evaluate.load("rouge")

# No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6
model_names = (
    "google/pegasus-cnn_dailymail",
    'sshleifer/distilbart-cnn-12-6'
)
rouge_results = []
for model_name in tqdm(model_names):
    pipe = pipeline(
        'summarization', model_name
    )
    
    res = rouge_metric.compute(
        predictions=pd.DataFrame(pipe(x)).squeeze().apply(lambda x: x.replace('<n>', ' ')).tolist(),
        references=y
    )
    rouge_results.append(res)
    
print(pd.DataFrame(rouge_results, index=model_names))

# cnn_dailymail
#                                  rouge1    rouge2    rougeL  rougeLsum
# google/pegasus-cnn_dailymail   0.447343  0.278494  0.367150   0.410628
# sshleifer/distilbart-cnn-12-6  0.475832  0.254762  0.403557   0.423165
    
# samsum
#                                  rouge1    rouge2    rougeL  rougeLsum
# google/pegasus-cnn_dailymail   0.150376  0.000000  0.126566   0.126566
# sshleifer/distilbart-cnn-12-6  0.305747  0.070197  0.220690   0.220690