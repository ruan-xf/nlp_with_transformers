import pandas as pd
from transformers import (
    pipeline, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
)


model_name = "google/pegasus-cnn_dailymail"
tokenizer=AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# type(model)
# <class 'transformers.models.pegasus.modeling_pegasus.PegasusModel'> 

models_info = []

for model in (
    AutoModel.from_pretrained(model_name),
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
):
    can_generate = model.can_generate()
    model_load_as = model.__class__.__name__
    models_info.append([can_generate, model_load_as])

    try:
        pipeline(
            'summarization',
            # model_name,
            model,
            tokenizer=tokenizer
        )
    except Exception as e:
        models_info[-1].append(e)


pd.DataFrame(models_info, columns=['can_generate', 'model_load_as', 'err']).to_clipboard()
# 	can_generate	model_load_as	err
# 0	False	PegasusModel	'SummarizationPipeline' object has no attribute 'assistant_model'
# 1	True	PegasusForConditionalGeneration	
