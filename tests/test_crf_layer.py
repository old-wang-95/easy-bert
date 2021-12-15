import unittest

import torch

from easy_bert.losses.crf_layer import CRF

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.crf = CRF(4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.crf.to(self.device)

    def test_decode(self):
        print('test_decode~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logits = torch.randn((2, 3, 4), requires_grad=False).to(self.device)  # (batch_size, seq_len, num_label)
        mask = torch.IntTensor([[1, 1, 1], [1, 1, 0]]).to(self.device)  # 第二个样本pad了一个位置
        labels, scores = self.crf.viterbi_decode(logits, mask)
        print(labels)

    def test_loss(self):
        print('test_loss~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logits = torch.randn((2, 3, 4), requires_grad=True).to(self.device)  # (batch_size, seq_len, num_label)
        mask = torch.IntTensor([[1, 1, 1], [1, 1, 0]]).to(self.device)  # 第二个样本pad了一个位置
        labels = torch.LongTensor([[0, 2, 3], [1, 0, 1]]).to(self.device)
        loss = self.crf.forward(logits, labels, mask)
        print(loss)


if __name__ == '__main__':
    unittest.main()
