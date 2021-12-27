目录
1. [标签平滑loss](#1-标签平滑loss)
2. [Focal loss](#2-focal-loss)
3. [crf loss](#3-crf-loss)

## 1. 标签平滑loss
标签平滑也是**一种正则化方法**，它**在label层面增加噪声**，使用soft label替代hard label。如下图：

<img height="300" src="images/label-smooth-img.png"/>

**交叉熵loss**定义为：

<img height="80" src="images/ce-loss.png"/>

其中：
- `K`为多分类的**类别数**，`y`为**onehot过后的label**：<img height="60" src="images/ce-loss-yi.png" align="center"/>
- `p`为logits经过**softmax后的概率**：<img height="60" src="images/ce-loss-pi.png" align="center"/>

**标签平滑使用soft label**，即：<img height="60" src="images/label-smooth-loss-yi.png" align="center"/>

- `α`为超参数，一般设置为`0.1`                      

## 2. Focal loss
在分类或序列标注任务中，可能存在**标签分布不均衡问题**，除了在数据层面做一些增强之外，还**可以换focal loss**。

回顾下**二元交叉熵**（BinaryCrossEntropyLoss）：

<img height="80" src="images/bce.png"/>

这里，
  - `y∈{±1}` 即**正确label**；
  - `p∈[0,1]` 即**模型预测label=1的概率**；

定义`p_t` <img height="60" src="images/bce-pt.png" align="center"/> ，于是 <img height="40" src="images/bce-2.png" align="center"/>

<br>

**交叉熵loss对不同的label是一视同仁的**， 对于NER，我们目标是让loss更多关注非“O”的标签（抓重点）

于是，可以**为每个类别设置一个独立的权重`α`**，即：

<img height="60" src="images/ce-loss-weight.png"/>

其中：`α∈[0,1]`，通常被**设置为label频率的倒数**，或者被**视为一个超参数**

<br>

尽管实现了重点优化非“O”标签，但可以更进一步，**重点优化那些较难的样本**。
可以为样本设置难度权重 `(1−p_t)^γ`，即

<img height="60" src="images/focal-loss.png"/>

- 越**简单的样本**，模型易将p预测接近`1`，其对loss贡献就少；
- 越**难的样本**，模型易将p预测接近`0`，其对loss贡献不变；

这里，`γ`**是一个超参数，控制降权程度**（对简单样本降权），如下图：

<img height="500" src="images/focal-loss-gama.png"/>

- `γ`**越大，对简单样本打压越厉害**；
- 当`γ`为`0`时，退化为交叉熵损失；

<br>

最终，我们综合得到focal loss：

<img height="60" src="images/focal-loss-1.png"/>

在交叉熵基础上，
- `α` 控制**重点优化那些频率较低的label**；
- `(1−p_t)^γ`控制**重点优化那些难学的样本**；

## 3. crf loss
crf层一般**被用在序列标注任务的最后一层**，**学习标签之间的转移**，参数为一个`n*n`的转移矩阵，`n`为label_size。下面是一个转移矩阵的例子：

<img height="200" src="images/crf-trans-matrix.png"/>

从图中可以看出，`b->s` `m->s`的转移几乎不太可能出现，因为权重很低。

**crf loss的目标**是：*让正确路径的score的比重*在*所有路径score之和*的占比(softmax)中越大越好。如下图：

<img height="300" src="images/viterbi.png"/>

我们目标是让 `B -> E -> B -> E` 这条**正确路径的score**占**所有路径的总score**的比例最大。

写成公式：

<img height="80" src="images/crf-loss-p.png"/>

正确路径我们一般称为gold path。使用负log最大似然，即得到**最终crf loss**：

<img height="80" src="images/crf-loss.png"/>