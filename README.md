# easy-bert
easy-bert是一个中文NLP工具，提供诸多**bert变体调用**和**调参方法**，**极速上手**；清晰的设计和代码注释，也**很适合学习**。

## 1. 极速上手
上手前，请**确保**：

1. 已从hugging face官网下载好[chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)，保存到某个目录，如：`./models/chinese-roberta-wwm-ext`；
2. 创建好你将要保存模型的目录，如：`/models/my_model`；

#### 分类任务
```python
from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer


pretrained_model_dir, your_model_idr = './models/chinese-roberta-wwm-ext', './models/my_model'
texts = ['天气真好', '今天运气很差']
labels = ['正面', '负面']

trainer = ClassificationTrainer(pretrained_model_dir, your_model_idr)
trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

predictor = ClassificationPredictor(pretrained_model_dir, your_model_idr)
labels = predictor.predict(texts)
```

#### 序列标注
```python
from easy_bert.bert4sequence_labeling.sequence_labeling_predictor import SequenceLabelingPredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer


pretrained_model_dir, your_model_idr = './models/chinese-roberta-wwm-ext', './models/my_model'
texts = [['你', '好', '呀'],['一', '马', '当', '先', '就', '是', '好']]
labels = [['B', 'E', 'S'],['B', 'M', 'M', 'E', 'S', 'S', 'S']]

trainer = SequenceLabelingTrainer(pretrained_model_dir, your_model_idr)
trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
        
predictor = SequenceLabelingPredictor(pretrained_model_dir, your_model_idr)
labels = predictor.predict(texts)
```

## 2. 调参指南
`Trainer`提供了丰富的参数可供选择
### 预训练模型
你可以快速替换预训练模型，即更改**pretrained_model_dir**参数，目前测试过的中文**预训练模型**包括：

- [albert_chinese_base](https://huggingface.co/voidful/albert_chinese_base)
- [chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm)
- [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)
- [bert-base-chinese](https://huggingface.co/bert-base-chinese)
- [chinese-electra-180g-base-discriminator](https://huggingface.co/hfl/chinese-electra-180g-base-discriminator)
- [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
- [TinyBERT_4L_zh](https://huggingface.co/huawei-noah/TinyBERT_4L_zh)
- [bert-distil-chinese](https://huggingface.co/adamlin/bert-distil-chinese)
- [longformer-chinese-base-4096](https://huggingface.co/schen/longformer-chinese-base-4096)

可以优先使用`chinese-roberta-wwm-ext`

### 学习率
bert微调一般使用较小的学习率`learning_rate`，如：`5e-5`, `3e-5`, `2e-5`

### 并行
可以为Trainer或Predictor设置`enable_parallel=True`，加速训练或推理。启用后，默认使用单机上的所有GPU。

### 对抗训练
对抗训练是一种正则化方法，主要是在embedding上加噪，缓解模型过拟合，默认`adversarial=None`，表示不对抗。

你可以设置：

- `adversarial='fgm'`：表示使用FGM对抗方法；
- `adversarial='pgd'`：表示使用PGD对抗方法；

### dropout_rate
dropout_rate随机丢弃一部分神经元来避免过拟合，隐含了集成学习的思想，默认`dropout_rate=0.5`

### loss
这里支持以下loss，通过`loss_type`参数来设置：

- `cross_entropy_loss`：标准的交叉熵loss，**`ClassificationTrainer`默认**；
- `label_smoothing_loss`：标签平滑loss，在label层面增加噪声，使用soft label替代hard label，**缓解过拟合**；
- `focal_loss`：**focal loss在类别不均衡时比较有用**，它允许为不同的label设置代价权重，并对简单的样本进行打压。
  - 你可以进一步设置`focal_loss_gamma`和`focal_loss_alpha`，默认`focal_loss_gamma=2` `focal_loss_alpha=None`
  - 设置`focal_loss_alpha`时，请**确保它是一个标签权重分布**，如：三分类设置`focal_loss_alpha=[1, 1, 1.5]`，表示我们更关注label_id为2的标签，因为它的样本数更少；
- `crf_loss`：**crf层学习标签与标签之间的转移**，仅支持序列标注任务，**`SequenceLabelingTrainer`默认**；

### 长文本
Bert的输入最多为512字，如果待处理的文本超过512字，你可以**截断**或者**分段**输入模型，也可以尝试Longformer模型：[longformer-chinese-base-4096](https://huggingface.co/schen/longformer-chinese-base-4096)，它使用稀疏自注意力，降低了自注意力的时空复杂度，将模型处理长度扩张到了`4096`

### 知识蒸馏
bert模型本身较重，资源受限下，想提高推理速度，知识蒸馏是一个不错的选择。

这里可以选择：

- `DistilBert`：是一个6层的Bert，预训练模型[bert-distil-chinese](https://huggingface.co/adamlin/bert-distil-chinese)在预训练阶段已经进行MLM任务的蒸馏，你可以**直接基于它进行下游任务的微调**；
  - 理论上，推理速度可以获得40%的提升，获得97%的bert-base效果
- `TinyBert`：[TinyBERT_4L_zh](https://huggingface.co/huawei-noah/TinyBERT_4L_zh)拥有4层、312的hidden_size，一般使用两阶段蒸馏，即下游任务也要蒸馏，可以使用`TinyBertDistiller`实现；
  - 理论上，4层的TinyBert，能够达到老师（Bert-base）效果的96.8%、参数量缩减为原来的13.3%、仅需要原来10.6%的推理时间

