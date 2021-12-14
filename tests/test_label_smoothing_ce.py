import unittest

import torch

from easy_bert.losses.label_smoothing_loss import LabelSmoothingCrossEntropy


class MyTestCase(unittest.TestCase):
    def test(self):
        print('test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        lsce = LabelSmoothingCrossEntropy()
        logits = torch.randn(4, 2)  # (batch_size=4, label_size=2)
        target = torch.tensor([0, 1, 1, 0])
        loss = lsce(logits, target)
        print(loss)

    def test_ignore_index(self):
        print('test_ignore_index~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        lsce = LabelSmoothingCrossEntropy(ignore_index=-1)
        logits = torch.randn(6, 2)  # (seq_len=4, label_size=2)
        target = torch.tensor([-1, 0, 1, 1, 0, -1])  # 序列标注一般首尾，即[CLS][SEP]部分用-1填充，计算loss时忽略它们
        loss = lsce(logits, target)
        print(loss)

    def test_reduction(self):
        print('test_reduction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        lsce = LabelSmoothingCrossEntropy(reduction='sum')
        logits = torch.randn(4, 2)  # (batch_size=4, label_size=2)
        target = torch.tensor([0, 1, 1, 0])
        loss = lsce(logits, target)
        print(loss)


if __name__ == '__main__':
    unittest.main()
