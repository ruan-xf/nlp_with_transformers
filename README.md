
项目整体说明：

这是我个人使用transformer模型进行nlp任务的学习项目，主要内容来自：
- [【NLP最佳实践】Huggingface Transformers实战教程 - Heywhale.com](https://www.heywhale.com/home/competition/61dd2a3dc238c000186ac330/content/0)

包括：
01-认识transformers
02-文本分类实战：基于Bert的企业隐患排查分类模型
03-文本多标签分类实战：基于Bert对推特文本进行多标签分类
04-句子相似性识别实战：基于Bert对句子对进行相似性二分类
05-命名实体识别实战：基于Bert实现文本NER任务
06-多项选择任务实战：基于Bert实现SWAG常识问题的多项选择
07-文本生成实战：基于预训练模型实现文本文本生成
08-文本摘要实战：基于Bert实现文本摘要任务
09-文本翻译实战：基于Bert实现端到端的机器翻译
10-问答实战：基于预训练模型实现QA

其中
- 当时不理解用法的，我写了小段的测试代码观察现象
- 繁琐的，会考虑使用其他的现成库，比如pytorch lightning
- 留下了外部资料、个人理解及做法
所以可以看作我对原项目的注解和拓展


前半部分（05及之前），原项目使用pytorch的，我疲于重复pytorch的训练和测试流程，改用了pytorch lightning

在不断的尝试中，我对这个库的理解也在加深，已不止于最初训练和测试流程的简化，还包括：
- torchmetrics的使用，提供了现成的指标计算方法
- 超参数的设置、访问和保存，包括trainer的参数
- 混合精度、梯度裁切的设置
- 数据加载和准备的分离

后半部分则为huggingface transformers库，虽然前半部分也是用transformers加载的预训练模型，但在这部分进一步使用huggingface生态也提供了相当的便利：
- Datasets，可以快速加载huggingface社区的数据集
- 使用DataCollator对数据集进行编码、掩码填充等预处理操作
- 较之pytorch lightning的进一步封装的Trainer类，针对下游任务的训练更方便，需要自己实现的步骤更少
- 使用pipeline快速加载已训练的模型进行预测，免于许多输入输出转换的步骤


