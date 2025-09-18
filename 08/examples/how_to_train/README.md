

Seq2SeqTrainer与普通Trainer的区别

来自[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) ：
```
The Trainer class is optimized for 🤗 Transformers models and can have surprising behaviors when used with other models. When using it with your own model, make sure:

your model always return tuples or subclasses of ModelOutput
your model can compute the loss if a labels argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
your model can accept multiple label arguments (use label_names in TrainingArguments to indicate their name to the Trainer) but none of them should be named "label"
```

其中
- loss是由model返回的，而不是由trainer计算
- 标签的移位在DataCollatorForSeq2Seq就完成了，与trainer无关
- BLEU、ROUGE指标也是自己在compute_metrics中定义的，Seq2SeqTrainer没内置


