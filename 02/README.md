
对原做法进行了改动：
- 相关库的版本更新
- 使用pytorch-lightning


acc的计算使用torchmetrics提供的accuracy，简化了`y_hat=max(logits, dim=1) -> (y_hat == y) / n`的步骤

import AdamW from
transformers -> torch.optim

pad_to_max_length=True ->
```
padding='max_length',
truncation=True,
```

