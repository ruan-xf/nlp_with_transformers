# hide_output


# Support for load_metric has been removed in datasets@3.0.0
# from datasets import load_metric
# bleu_metric = load_metric("sacrebleu")


import evaluate

bleu_metric = evaluate.load('sacrebleu')

import pandas as pd
import numpy as np



# 原来reference是list，改了也可以
bleu_metric.add(
    # prediction="the the the the the the",
    prediction='the',
    reference="the cat is on the mat"
)
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
pd.DataFrame.from_dict(results, orient="index", columns=["Value"])