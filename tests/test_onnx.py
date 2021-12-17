import time
import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model_dir = './tests/test_model'
        self.pretrained_model_dir = './models/chinese-roberta-wwm-ext'

    def test_classification(self):
        print('test_classification~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = ['天气真好', '今天运气很差']
        labels = ['正面', '负面']
        trainer = ClassificationTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        # 测试torch模型速度
        predictor = ClassificationPredictor(self.pretrained_model_dir, self.model_dir)
        init_time = time.time()
        for _ in range(10):
            predictor.predict(texts)
        print('torch model cost time: {}s'.format(time.time() - init_time))

        # 测试onnx模型速度
        predictor.transform2onnx()
        predictor.predict(texts)  # 先预热一次（否则结果可能误导你）
        init_time = time.time()
        for _ in range(10):
            predictor.predict(texts)
        print('onnx model cost time: {}s'.format(time.time() - init_time))


if __name__ == '__main__':
    unittest.main()
