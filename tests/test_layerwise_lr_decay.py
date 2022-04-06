import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.bert4sequence_labeling.sequence_labeling_predictor import SequenceLabelingPredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer


class MyTestCase(unittest.TestCase):
    def test_bert2classification(self):
        print('test_bert2classification~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir, model_dir = './models/chinese-roberta-wwm-ext', './tests/test_model'
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(
            pretrained_model_dir, model_dir, layer_wise_lr_decay=True, lr_decay_rate=0.95
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        predictor = ClassificationPredictor(pretrained_model_dir, model_dir)
        labels = predictor.predict(texts)
        print(labels)

    def test_bert2sequence_labeling(self):
        print('test_bert2sequence_labeling~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir, model_dir = './models/chinese-roberta-wwm-ext', './tests/test_model'
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = [
            ['B', 'E', 'S'],
            ['B', 'M', 'M', 'E', 'S', 'S', 'S']
        ]
        trainer = SequenceLabelingTrainer(
            pretrained_model_dir, model_dir, layer_wise_lr_decay=True, lr_decay_rate=0.95
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
        predictor = SequenceLabelingPredictor(pretrained_model_dir, model_dir)
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
