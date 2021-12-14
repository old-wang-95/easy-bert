import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = './tests/test_model'
        # 经测试过的中文bert模型
        self.support_models = [
            'albert_chinese_base', 'chinese-bert-wwm', 'chinese-macbert-base', 'bert-base-chinese',
            'chinese-electra-180g-base-discriminator', 'chinese-roberta-wwm-ext', 'TinyBERT_4L_zh',
            'bert-distil-chinese', 'longformer-chinese-base-4096'
        ]

    def test_general_bert(self):
        print('test_general_bert~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']
        for pretrained_model_dir in self.support_models:
            pretrained_model_dir = './models/{}'.format(pretrained_model_dir)
            print('current test model:', pretrained_model_dir)
            trainer = ClassificationTrainer(pretrained_model_dir, self.model_dir, learning_rate=5e-5)
            trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
            print('model {} test success'.format(pretrained_model_dir))

    def test_enable_parallel(self):
        print('test_enable_parallel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir = './models/chinese-roberta-wwm-ext'
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(pretrained_model_dir, self.model_dir, learning_rate=5e-5, enable_parallel=True)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

    def test_loss_type(self):
        print('test_loss_type~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir = './models/chinese-roberta-wwm-ext'
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(
            pretrained_model_dir, self.model_dir, learning_rate=5e-5,
            loss_type='focal_loss', focal_loss_gamma=2, focal_loss_alpha=[1, 1]
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        trainer = ClassificationTrainer(
            pretrained_model_dir, self.model_dir, learning_rate=5e-5, loss_type='label_smoothing_loss'
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

    def test_adversarial(self):
        print('test_adversarial~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir = './models/chinese-roberta-wwm-ext'
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(pretrained_model_dir, self.model_dir, learning_rate=5e-5, adversarial='fgm')
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        trainer = ClassificationTrainer(pretrained_model_dir, self.model_dir, learning_rate=5e-5, adversarial='pgd')
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

    def test_predictor(self):
        print('test_predictor~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pretrained_model_dir = './models/chinese-roberta-wwm-ext'
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        trainer = ClassificationTrainer(
            pretrained_model_dir, self.model_dir, learning_rate=5e-5, enable_parallel=True
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)
        predictor = ClassificationPredictor(pretrained_model_dir, self.model_dir, enable_parallel=True)
        texts = [
            '天气真好',
            '今天运气很差',
            '天气不错',
            '今天运气很烂哦',
        ]
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
