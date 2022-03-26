import unittest

from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.bert4pretraining.mlm_trainer import MaskedLMTrainer
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pretrained_model_dir = './models/chinese-roberta-wwm-ext'
        self.model_dir = './tests/test_model'

    def test_bert4classification(self):
        print('test_bert4classification~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = ['天气真好', '今天运气很差']
        labels = ['正面', '负面']
        # 第一次训练
        trainer = ClassificationTrainer(self.pretrained_model_dir, self.model_dir)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
        # 继续训练
        trainer = ClassificationTrainer(self.pretrained_model_dir, self.model_dir, load_last_ckpt=True)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=10)

    def test_bert4sequence_labeling(self):
        print('test_bert4sequence_labeling~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = [
            ['B', 'E', 'S'],
            ['B', 'M', 'M', 'E', 'S', 'S', 'S']
        ]
        # 第一次训练
        trainer = SequenceLabelingTrainer(self.pretrained_model_dir, self.model_dir)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
        # 继续训练
        trainer = SequenceLabelingTrainer(self.pretrained_model_dir, self.model_dir, load_last_ckpt=True)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=10)

    def test_bert4pretraining(self):
        print('test_bert4pretraining~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            '早上起床后，我发现今天天气还真是不错的。早上起床后，我发现今天天气还真是不错的。'
        ]
        # 第一次训练
        trainer = MaskedLMTrainer(self.pretrained_model_dir, self.model_dir)
        trainer.train(texts, batch_size=1, epoch=10, saved_step=5)
        # 继续训练
        trainer = MaskedLMTrainer(self.pretrained_model_dir, self.model_dir, load_last_ckpt=True)
        trainer.train(texts, batch_size=1, epoch=5, saved_step=5)


if __name__ == '__main__':
    unittest.main()
