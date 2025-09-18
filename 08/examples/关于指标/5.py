import evaluate

bleu_metric = evaluate.load('sacrebleu')

import pandas as pd
import numpy as np




bleu_metric.add(
    # prediction="the the the the the the",
    prediction='the',
    reference=["the cat is on the mat"]
)
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
pd.DataFrame.from_dict(results, orient="index", columns=["Value"])

# Value
# score	57.893007
# counts	[5, 3, 2, 1]
# totals	[5, 4, 3, 2]
# precisions	[100.0, 75.0, 66.67, 50.0]
# bp	0.818731
# sys_len	5
# ref_len	6


# 结果解释
# 好的，这是一个非常典型的BLEU分数计算示例。我们来逐项解释这个结果：

# 这个BLEU分数计算是基于：
# *   **预测句子（Prediction）**: "the cat is on mat" （5个词）
# *   **参考译文（Reference）**: ["the cat is on the mat"] （6个词）

# ### 结果解读

# | 指标 | 值 | 解释 |
# | :--- | :--- | :--- |
# | **`score`** | **57.89** | **这是最终的BLEU分数，是核心指标。** 它表示机器翻译（预测句子）与参考译文之间的相似度。57.89可以粗略地理解为**57.89%** 的匹配度。这个分数相当高，因为预测句子和参考译文只差了一个词。 |
# | **`counts`** | [5, 3, 2, 1] | 这是匹配到的 **n-gram** 的数量。 <br> 1-gram（单词）： 预测句有5个词（'the', 'cat', 'is', 'on', 'mat'），它们全部在参考译文中出现，所以是 **5**。 <br> 2-gram（词组）： 预测句有4个2-gram（'the cat', 'cat is', 'is on', 'on mat'），其中前3个在参考译文中出现（'on mat' 不匹配，参考译文中是 'on the'），所以是 **3**。 <br> 3-gram： 预测句有3个3-gram（'the cat is', 'cat is on', 'is on mat'），其中前2个匹配（'is on mat' 不匹配），所以是 **2**。 <br> 4-gram： 预测句有2个4-gram（'the cat is on', 'cat is on mat'），只有第1个匹配，所以是 **1**。 |
# | **`totals`** | [5, 4, 3, 2] | 这是预测句子中 **n-gram 的总数**。 <br> 1-gram总数：5个词 -> **5** <br> 2-gram总数：4个 -> **4** <br> 3-gram总数：3个 -> **3** <br> 4-gram总数：2个 -> **2** |
# | **`precisions`** | [100.0, 75.0, 66.67, 50.0] | 这是 **n-gram 精确度**，是 `counts[n] / totals[n]` 的结果，是BLEU分数的核心组成部分。 <br> 1-gram精度：5/5 = **100%** <br> 2-gram精度：3/4 = **75%** <br> 3-gram精度：2/3 ≈ **66.67%** <br> 4-gram精度：1/2 = **50%** <br> **BLEU分数就是这些精度的几何平均值**（再乘以 brevity penalty）。 |
# | **`bp`** | 0.818731 | ** brevity penalty（过短惩罚）**。因为预测句子长度（5个词）比参考译文长度（6个词）要短，所以需要施加惩罚。计算公式是 `exp(1 - ref_len/sys_len)`。这里 `exp(1 - 6/5) = exp(-0.2) ≈ 0.8187`。这个因子会乘以精度平均值，以防止输出过短的翻译也能得高分。 |
# | **`sys_len`** | 5 | 预测句子（System）的长度（词数）。 |
# | **`ref_len`** | 6 | 参考译文（Reference）的长度（词数）。 |

# ### 最终分数是如何计算的？

# 1.  计算几何平均精度： `geometric_mean = (1.0 * 0.75 * 0.6667 * 0.5) ^ (1/4) ≈ (0.2500) ^ (0.25) ≈ 0.7071`
# 2.  乘以过短惩罚： `0.7071 * 0.8187 ≈ 0.5789`
# 3.  转换为百分比形式（并四舍五入）： `0.5789 * 100 = 57.89`

# **总结**：预测句子 "the cat is on mat" 与参考译文 "the cat is on the mat" 非常相似，只少了一个 "the"，因此获得了较高的BLEU分数（57.89）。这个分数由高精度的1-gram、2-gram和适度的过短惩罚共同决定。




# 几何平均数的计算
data = [100.0, 75.0, 66.67, 50.0]


# 
# np.exp(np.mean(np.log(data))) 计算的是几何平均数，原因在于它通过以下步骤还原了几何平均的本质：

# 1. **对数变换**：`np.log(data)` 对数据取自然对数，将乘法关系转化为加法关系（即 $\log(ab) = \log a + \log b$）；
# 2. **算术平均**：`np.mean` 对对数后的值求算术平均，相当于计算乘积的 $n$ 次方根的指数部分（$\frac{1}{n}\sum \log x_i = \log(\prod x_i)^{1/n}$）；
# 3. **指数还原**：`np.exp` 将结果转换回原始尺度，最终得到 $\exp(\log(\prod x_i)^{1/n}) = (\prod x_i)^{1/n}$，即几何平均的定义。

# 因此，该表达式是几何平均数的高效数值计算方法，尤其适合处理可能溢出的连乘积运算。
import numpy as np
gm = np.exp(np.mean(np.log(data)))


# 
import scipy
gm = scipy.stats.gmean(data)


# .7071*0.818731 = 0.5789246900999999, 完成验证