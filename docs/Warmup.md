目录
1. [Warmup原理](#1-warmup原理)
2. [实践](#2-实践)

## 1. Warmup原理
warmup使用**动态的学习率**（一般lr先增大 后减小），
- lr一开始别太大，有助于缓解模型在初始阶段，对前几个batch数据过拟合；
  - 训练初期，模型对数据还比较陌生，较大的学习率可能会破坏预训练好的权重
- lr后面小一点，有助于模型后期的稳定；
  - 训练稳定期，较大的学习率可能会破坏模型的稳定，导致跳出最优解

常见的**warmup种类**：
  - **constant**，表示使用恒定学习率，lr曲线为 <img src="./images/constant_warmup.png" width=500 align="center">
  - **cosine**，表示余弦曲线学习率，lr曲线为 <img src="./images/cosine_warmup.png" width=500 align="center">
  - **linear**，表示线性学习率，lr曲线为 <img src="./images/linear_warmup.png" width=500 align="center">

## 2. 实践
transformers提供了上述几种warmup，可以直接使用。

首先**设置warmup**，代码示例：
```python
# 根据warmup配置，设置warmup
total_steps = len(train_texts) // batch_size * epoch
num_warmup_steps = self.warmup_step_num if isinstance(self.warmup_step_num, int) else \
    int(total_steps * self.warmup_step_num)
assert num_warmup_steps <= total_steps, \
    'num_warmup_steps {} is too large, more than total_steps {}'.format(num_warmup_steps, total_steps)
if self.warmup_type == 'linear':
    warmup_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, total_steps)
elif self.warmup_type == 'cosine':
    warmup_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, total_steps)
else:
    warmup_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
```
然后**在每个训练step内**，执行`warmup_scheduler.step()`，不断**更新lr**；

**完整代码**请参考以下源代码：
- [easy_bert/bert4classification/classification_trainer.py](https://github.com/waking95/easy-bert/blob/main/easy_bert/bert4classification/classification_trainer.py)
- [easy_bert/bert4sequence_labeling/sequence_labeling_trainer.py](https://github.com/waking95/easy-bert/blob/main/easy_bert/bert4sequence_labeling/sequence_labeling_trainer.py)
