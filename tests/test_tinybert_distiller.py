import unittest

from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.bert4sequence_labeling.sequence_labeling_predictor import SequenceLabelingPredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer
from easy_bert.tinybert_distiller import TinyBertDistiller


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.teacher_pretrained = './models/chinese-roberta-wwm-ext'
        self.teacher_model_dir = './tests/test_model'
        self.student_pretrained = './models/TinyBERT_4L_zh'
        self.student_model_dir = './tests/test_model2'

    def test_general(self):
        print('test_classification~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            '天气真好',
            '今天运气很差',
        ]
        labels = ['正面', '负面']

        # 训练老师模型
        trainer = ClassificationTrainer(self.teacher_pretrained, self.teacher_model_dir)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        # 蒸馏学生
        distiller = TinyBertDistiller(
            self.teacher_pretrained, self.teacher_model_dir, self.student_pretrained, self.student_model_dir,
            task='classification'
        )
        distiller.distill_train(texts, labels, max_len=20, epoch=20, batch_size=2)

        # 加载fine-tune蒸馏过的模型
        predictor = ClassificationPredictor(self.student_pretrained, self.student_model_dir)
        print(predictor.predict(texts))

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

        # 训练老师模型
        trainer = SequenceLabelingTrainer(self.teacher_pretrained, self.teacher_model_dir, loss_type='crf_loss')
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

        # 蒸馏学生
        distiller = TinyBertDistiller(
            self.teacher_pretrained, self.teacher_model_dir, self.student_pretrained, self.student_model_dir,
            task='sequence_labeling', hard_label_loss='crf_loss'
        )
        distiller.distill_train(texts, labels, max_len=20, epoch=20, batch_size=2)

        # 加载fine-tune蒸馏过的模型
        predictor = SequenceLabelingPredictor(self.student_pretrained, self.student_model_dir)
        print(predictor.predict(texts))


if __name__ == '__main__':
    unittest.main()
