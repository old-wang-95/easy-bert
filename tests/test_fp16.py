import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model_dir = './tests/test_model'
        self.pretrained_model_dir = './models/chinese-roberta-wwm-ext'

    def test_classification(self):
        print('test_classification~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = ['天气真好', '今天运气很差']
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5, enable_fp16=True)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        predictor = ClassificationPredictor(self.pretrained_model_dir, self.model_dir, enable_fp16=True)
        print(predictor.predict(texts))


if __name__ == '__main__':
    unittest.main()
