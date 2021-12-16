import os
import re
from functools import partial

from datasets.arrow_dataset import Dataset
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW

from easy_bert import logger
from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.bert4sequence_labeling.sequence_labeling_predictor import SequenceLabelingPredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_trainer import SequenceLabelingTrainer


class TinyBertDistiller(object):
    """
    TinyBert蒸馏器，用12层的bert-base蒸馏4层的TinyBert
    """

    def __init__(self, teacher_pretrained_dir, teacher_model_idr, student_pretrained_dir, student_model_dir,
                 task='classification', enable_parallel=False,
                 hard_label_loss='cross_entropy_loss', focal_loss_gamma=2, focal_loss_alpha=None,
                 temperature=4, hard_label_weight=1,
                 kd_loss_type='ce', kd_loss_weight=1.2, lr=1e-4, ckpt_frequency=1):
        # 初始化目录
        self.teacher_pretrained_dir, self.teacher_model_idr = teacher_pretrained_dir, teacher_model_idr
        self.student_pretrained_dir, self.student_model_dir = student_pretrained_dir, student_model_dir

        # 初始化task
        assert task in ('classification', 'sequence_labeling')
        self.task = task
        self.enable_parallel = enable_parallel

        # 初始化学生hard label的loss_type，即对真实label的loss
        self.hard_label_loss = hard_label_loss
        self.focal_loss_gamma, self.focal_loss_alpha = focal_loss_gamma, focal_loss_alpha

        # 初始化蒸馏配置
        self.temperature = temperature
        self.hard_label_weight = hard_label_weight
        self.kd_loss_type, self.kd_loss_weight = kd_loss_type, kd_loss_weight
        self.lr = lr
        self.ckpt_frequency = ckpt_frequency

        # 分别加载老师model和学生model
        self.teacher_model, self.teacher_predictor = self._load_teacher_model()
        self.student_model, self.student_trainer = self._load_student_model()

    def _load_teacher_model(self):
        """加载老师模型，通过Predictor获取老师模型"""
        logger.info('start to load teacher model ...')
        mapper = {'classification': ClassificationPredictor, 'sequence_labeling': SequenceLabelingPredictor}
        teacher_predictor = mapper[self.task](
            self.teacher_pretrained_dir, self.teacher_model_idr, enable_parallel=self.enable_parallel
        )
        teacher_model = teacher_predictor.model
        teacher_model.forward = partial(teacher_model.forward, return_extra=True)  # 启用return_extra
        logger.info('load teacher model success')
        return teacher_model, teacher_predictor

    def _load_student_model(self):
        """加载学生模型，通过Trainer构建学生模型"""
        logger.info('start to load student model ...')
        mapper = {'classification': ClassificationTrainer, 'sequence_labeling': SequenceLabelingTrainer}
        student_trainer = mapper[self.task](
            self.student_pretrained_dir, self.student_model_dir, learning_rate=self.lr,
            enable_parallel=self.enable_parallel, loss_type=self.hard_label_loss,
            focal_loss_alpha=self.focal_loss_alpha, focal_loss_gamma=self.focal_loss_gamma
        )
        student_trainer.vocab = self.teacher_predictor.vocab  # 学生和老师使用相同的vocab
        student_trainer._build_model()  # 构建模型
        student_trainer.vocab.save_vocab('{}/{}'.format(student_trainer.model_dir, student_trainer.vocab_name))
        student_trainer._save_config()  # 保存训练config
        student_model = student_trainer.model
        student_model.forward = partial(student_model.forward, return_extra=True)  # 启用return_extra
        logger.info('student model load success')
        return student_model, student_trainer

    def distill_train(self, texts, labels, max_len=256, epoch=20, batch_size=32):
        """
        蒸馏训练
        :param texts: list[str] or list[list[str]]
        :param labels: list[str] or list[list[str]]
        :param max_len: 最大长度，超过将被截断
        :param epoch: 迭代轮数，蒸馏一般epoch较大，20~50
        :param batch_size:
        """
        # 封装dataset和dataloader，适配textbrewer接口
        train_input_ids, train_att_mask, train_label_ids = self.student_trainer._transform_batch(
            texts, labels, max_length=max_len
        )
        train_dataset = Dataset.from_dict(
            {'input_ids': train_input_ids, 'attention_mask': train_att_mask, 'labels': train_label_ids}
        )
        train_dataset.set_format(type='torch')  # 设置data type为torch的tensor
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # 蒸馏配置
        distill_config = DistillationConfig(
            # 设置温度系数temperature, tiny-bert论文作者使用1表现最好，一般大于1比较好
            temperature=self.temperature,
            # 设置ground truth loss权重
            hard_label_weight=self.hard_label_weight,
            # 设置预测层蒸馏loss（即soft label损失）为交叉熵，并稍微放大其权重
            kd_loss_type=self.kd_loss_type, kd_loss_weight=self.kd_loss_weight,
            # 配置中间层蒸馏映射
            intermediate_matches=[
                # 配置hidden蒸馏映射、维度映射
                {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
                 'proj': ['linear', 312, 768]},  # embedding层输出
                {'layer_T': 3, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
                 'proj': ['linear', 312, 768]},
                {'layer_T': 6, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
                 'proj': ['linear', 312, 768]},
                {'layer_T': 9, 'layer_S': 3, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
                 'proj': ['linear', 312, 768]},
                {'layer_T': 12, 'layer_S': 4, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
                 'proj': ['linear', 312, 768]},
                # 配置attention矩阵蒸馏映射，注意layer序号从0开始
                {"layer_T": 2, "layer_S": 0, "feature": "attention", "loss": "attention_mse", "weight": 1},
                {"layer_T": 5, "layer_S": 1, "feature": "attention", "loss": "attention_mse", "weight": 1},
                {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_mse", "weight": 1},
                {"layer_T": 11, "layer_S": 3, "feature": "attention", "loss": "attention_mse", "weight": 1},
            ]
        )

        # 训练配置
        optimizer = AdamW(self.student_model.parameters(), lr=self.lr)  # 使用大一点的lr
        train_config = TrainingConfig(
            output_dir=self.student_model_dir, device=self.student_trainer.device,
            data_parallel=self.enable_parallel, ckpt_frequency=self.ckpt_frequency  # 一个epoch存ckpt_frequency次模型
        )

        # 配置model中logits hiddens attentions losses的获取方法
        def simple_adaptor(batch, model_outputs):
            return {
                'logits': model_outputs[-1]['logits'], 'hidden': model_outputs[-1]['hiddens'],
                'attention': model_outputs[-1]['attentions'], 'losses': model_outputs[1],
            }

        # 蒸馏
        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=self.teacher_model, model_S=self.student_model,
            adaptor_T=simple_adaptor, adaptor_S=simple_adaptor
        )
        with distiller:
            logger.info('start to knowledge distill ...')
            distiller.train(optimizer, train_dataloader, num_epochs=epoch)
            logger.info('distill finish')

        # 重命名文件名
        newest_model_name = sorted(  # 根据textbrewer模型文件格式，查找最新的模型
            [f for f in os.listdir(self.student_model_dir) if 'pkl' in f],
            key=lambda f: int(re.findall('\\d+', f)[0]), reverse=True
        )[0]
        model_path, new_model_path = \
            '{}/{}'.format(self.student_model_dir, newest_model_name), '{}/bert_model.bin'.format(
                self.student_model_dir)
        logger.info('copy model {} to {}'.format(model_path, new_model_path))
        os.system('cp {} {}'.format(model_path, new_model_path))
        logger.info('now, you can load {} with Predictor'.format(self.student_model_dir))
