
我发现原notebook中出现的这个gpt2-xl特别慢，担心其他模型也这么慢就没得玩了，所以
- gpt2-xl
- t5-large
- facebook/bart-large-cnn
- google/pegasus-cnn_dailymail

以上模型都加载来进行推理，计时，并且在超过30分钟没有结果就中断

是在google colab中进行的，因为：
- ai给的测试代码用到了linux的系统api
- 下载速度超级快
- 加载模型至少需要3个G的内存吧，也不用在我的机器上跑


万幸，只有gpt2-xl会超时：
- GPT-2-xl timed out after 30 minutes
- t5: 52.50 seconds
- bart: 28.11 seconds
- pegasus: 173.38 seconds
