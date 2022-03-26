import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer


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
            pretrained_model_dir, model_dir,
            add_on='bilstm', rnn_hidden=256, rnn_lr=1e-3
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        predictor = ClassificationPredictor(pretrained_model_dir, model_dir)
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
