import unittest

import torch

from easy_bert.losses.label_smoothing_loss import LabelSmoothingCrossEntropy


class MyTestCase(unittest.TestCase):
    def test_something(self):
        lsce = LabelSmoothingCrossEntropy()
        logits = torch.randn(4, 2)  # (batch_size=4, label_size=2)
        target = torch.tensor([0, 1, 1, 0])
        loss = lsce(logits, target)
        print(loss)


if __name__ == '__main__':
    unittest.main()
