import time
import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.bert4sequence_labeling.sequence_labeling_predictor import SequenceLabelingPredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer


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

    def test_sequence_labeling(self):
        print('test_sequence_labeling~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = [
            ['B', 'E', 'S'],
            ['B', 'M', 'M', 'E', 'S', 'S', 'S']
        ]
        trainer = SequenceLabelingTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5,
                                          loss_type='cross_entropy_loss')  # crf层转onnx暂时有点问题
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        # 测试torch模型速度
        predictor = SequenceLabelingPredictor(self.pretrained_model_dir, self.model_dir)
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
