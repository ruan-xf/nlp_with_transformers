

transformers版本
在我当前的版本，The pad_to_max_length argument is deprecated 

配置了作者的环境比较差异

python==3.9
transformers==4.17.0

在保留其他参数设置的情况下，
```
padding='max_length',
truncation=True,
```
与`pad_to_max_length=True`等效



涉及的指标，不知道为什么要加的扁平化处理
- [Multilabel(多标签分类)metrics：hamming loss，F score - fledglingbird - 博客园](https://www.cnblogs.com/fledlingbird/p/10675922.html)
- [深入理解Micro-F1：多标签分类评估指标详解_芝士AI吃鱼的技术博客_51CTO博客](https://blog.51cto.com/u_15610758/12368489)
