import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑过的交叉熵loss"""

    def __init__(self, alpha=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha

        assert reduction in ('mean', 'sum')
        self.reduction = reduction  # 对batch维度的loss的处理策略，可以是mean或者sum

        self.ignore_index = ignore_index  # 忽略target中为ignore_index的位置

    def forward(self, logits, target):
        """计算标签平滑过的交叉熵loss，请参照公式"""

        # 获取类别数K
        K = logits.size()[-1]

        # 对logits计算softmax得到概率，并取log
        log_preds = F.log_softmax(logits, dim=-1)
        # 去掉ignore_index位置的结果
        active_log_preds = log_preds[target != self.ignore_index]

        # 计算非target部分的loss
        if self.reduction == 'sum':
            no_target_loss = -active_log_preds.sum()  # 同时沿着batch和label维度求和
        else:
            no_target_loss = -active_log_preds.sum(dim=-1)  # 沿着label维度求和
            if self.reduction == 'mean':
                no_target_loss = no_target_loss.mean()  # 沿着batch维度平均
        no_target_loss = self.alpha / K * no_target_loss  # 乘上非target部分的yi

        # 计算target部分的loss
        # 借助torch自带的计算交叉熵的函数F.nll_loss，并乘上target部分的yi
        target_loss = (1 - self.alpha) * F.nll_loss(log_preds, target,
                                                    reduction=self.reduction, ignore_index=self.ignore_index)

        return no_target_loss + target_loss
