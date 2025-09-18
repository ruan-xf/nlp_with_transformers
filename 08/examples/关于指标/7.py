

predicts = ["usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow . the 26-year-old sprinter claims his third gold at the world championships in moscow . he charges clear of united states rival Justin gatlin to win in 37.36 seconds .",
 "Usain Bolt anchored Jamaica to victory in the men's 4x100mrelay. The Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, NickelAshmeade and Bolt won in 37.36 seconds. The 26-year-old Bolt has now won three golds at the world championships.",
 "Usain Bolt wins third gold of world championships .<n>Anchors Jamaica to victory in men's 4x100m relay .<n>U.S. finishes second with Canada taking the bronze ."]


predicts[-1] = predicts[-1].replace('.<n>', '.\n')
# print(predicts[-1])

import evaluate
import numpy as np
import pandas as pd

bleu_metric = evaluate.load('sacrebleu')
# ImportError: To be able to use evaluate-metric/rouge, you need to install the following dependencies['nltk', 'rouge_score'] using 'pip install # Here to have a nice missing dependency error message early on rouge_score' for instance'
rouge_metric = evaluate.load("rouge")

def bleu(pred: str, target: str):
    bleu_metric.add(
        prediction=pred,
        reference=target
    )
    results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
    results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
    return pd.DataFrame.from_dict(results, orient="index", columns=["Value"])




# 观察正常使用标点、换行是否会影响n-gram的计数，从而导致不同的bleu得分
target = '''Usain Bolt wins third gold of world championship .
Anchors Jamaica to 4x100m relay victory .
Eighth gold at the championships for Bolt .
Jamaica double up in women's 4x100m relay .'''

pred1 = "usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow . the 26-year-old sprinter claims his third gold at the world championships in moscow . he charges clear of united states rival Justin gatlin to win in 37.36 seconds ."
pred2 = '''usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow . 
the 26-year-old sprinter claims his third gold at the world championships in moscow . 
he charges clear of united states rival Justin gatlin to win in 37.36 seconds .'''
pred3 = '''usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow .
the 26-year-old sprinter claims his third gold at the world championships in moscow .
he charges clear of united states rival Justin gatlin to win in 37.36 seconds .'''
pred4 = '''usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow.
the 26-year-old sprinter claims his third gold at the world championships in moscow.
he charges clear of united states rival Justin gatlin to win in 37.36 seconds.'''


# 结果是完全一样的，但score为0是什么意思？
# bleu(pred1, target)

rouge_metric.add(
    prediction=predicts,
    reference=target
)
rouge_metric.compute()