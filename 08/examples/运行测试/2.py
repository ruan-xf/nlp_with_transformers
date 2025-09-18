import pandas as pd
from transformers import pipeline

sample_text = '''(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his
third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m
relay. The fastest man in the world charged clear of United States rival Justin
Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel
Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds
with Canada taking the bronze after Britain were disqualified for a faulty
handover. The 26-year-old Bolt has n
'''


# pipe = pipeline("text-generation", model="gpt2-xl")
# print(pipe(sample_text+'\nTL;DR:'))


# pipe = pipeline("summarization", model="t5-large")
# print(pipe(sample_text))


# pipe = pipeline("summarization", model="facebook/bart-large-cnn")
# print(pipe(sample_text))


# pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")
# print(pipe(sample_text))


# [SummarizationPipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.SummarizationPipeline)
# This summarizing pipeline can currently be loaded from pipeline() using the following task identifier: "summarization".
# Returns

# A list or a list of list of dict

# Each result comes as a dictionary with the following keys:

# summary_text (str, present when return_text=True) — The summary of the corresponding input.
# summary_token_ids (torch.Tensor or tf.Tensor, present when return_tensors=True) — The token ids of the summary.

# Summarize the text(s) given as inputs.

results = {'t5': ([{'summary_text': "usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow . the 26-year-old sprinter claims his third gold at the world championships in moscow . he charges clear of united states rival Justin gatlin to win in 37.36 seconds ."}],
  52.50412559509277),
 'bart': ([{'summary_text': "Usain Bolt anchored Jamaica to victory in the men's 4x100mrelay. The Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, NickelAshmeade and Bolt won in 37.36 seconds. The 26-year-old Bolt has now won three golds at the world championships."}],
  28.10715126991272),
 'pegasus': ([{'summary_text': "Usain Bolt wins third gold of world championships .<n>Anchors Jamaica to victory in men's 4x100m relay .<n>U.S. finishes second with Canada taking the bronze ."}],
  173.37881302833557)}


df = pd.DataFrame(results)
ser = df.iloc[0]

outs = ser.apply(lambda x: x[0]['summary_text']).tolist()

# ["usain bolt anchors Jamaica to victory in the men's 4x100m relay in Moscow . the 26-year-old sprinter claims his third gold at the world championships in moscow . he charges clear of united states rival Justin gatlin to win in 37.36 seconds .",
#  "Usain Bolt anchored Jamaica to victory in the men's 4x100mrelay. The Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, NickelAshmeade and Bolt won in 37.36 seconds. The 26-year-old Bolt has now won three golds at the world championships.",
#  "Usain Bolt wins third gold of world championships .<n>Anchors Jamaica to victory in men's 4x100m relay .<n>U.S. finishes second with Canada taking the bronze ."]


