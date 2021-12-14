import json
import math
import os
import random

import numpy as np
import torch
import torch.cuda
from sklearn.utils import shuffle
from torch.nn import DataParallel
from transformers import AdamW

from easy_bert import logger
from easy_bert.adversarial import FGM, PGD
from easy_bert.base.base_trainer import BaseTrainer
from easy_bert.bert4classification.classification_model import ClassificationModel
from easy_bert.vocab import Vocab


class ClassificationTrainer(BaseTrainer):
    def __init__(self, pretrained_model_dir, model_dir, learning_rate=5e-5, ckpt_name='bert_model.bin',
                 vocab_name='vocab.json', enable_parallel=False, adversarial=None, dropout_rate=0.5,
                 loss_type='cross_entropy_loss', focal_loss_gamma=2, focal_loss_alpha=None, random_seed=0):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name
        self.vocab_name = vocab_name
        self.enable_parallel = enable_parallel

        # 设置随机种子
        self.random_seed = random_seed
        self._set_random_seed(random_seed)

        self.adversarial = adversarial
        assert adversarial in (None, 'fgm', 'pgd')

        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        # 自动获取当前设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = None
        self.epoch = None

        self.vocab = Vocab()

    def _set_random_seed(self, seed):
        """针对torch torch.cuda numpy random分别设定随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self):
        """构建bert模型"""
        # 实例化ClassificationModel模型
        self.model = ClassificationModel(
            self.pretrained_model_dir, self.vocab.label_size, dropout_rate=self.dropout_rate,
            loss_type=self.loss_type, focal_loss_alpha=self.focal_loss_alpha, focal_loss_gamma=self.focal_loss_gamma
        )

        # 设置AdamW优化器
        no_decay = ["bias", "LayerNorm.weight"]  # bias和LayerNorm不使用正则化
        # 参数分两组，分为decay or no_decay
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # 使用bert的vocab更新Vocab对象
        self.vocab.set_vocab2id(self.model.get_bert_tokenizer().vocab)
        self.vocab.set_id2vocab({_id: char for char, _id in self.vocab.vocab2id.items()})
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])

        # 启用并行，使用DataParallel封装model
        if self.enable_parallel:
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # 将模型拷贝到当前设备
        self.model.to(self.device)

    def _save_config(self):
        """保存训练参数配置为json文件"""
        config = {
            'vocab_size': self.vocab.vocab_size,
            'label_size': self.vocab.label_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            'epoch': self.epoch,
            'ckpt_name': self.ckpt_name,
            'vocab_name': self.vocab_name,
            'enable_parallel': self.enable_parallel,
            'adversarial': self.adversarial,
            'loss_type': self.loss_type,
            'focal_loss_gamma': self.focal_loss_gamma,
            'focal_loss_alpha': self.focal_loss_alpha,
            'pretrained_model': os.path.basename(self.pretrained_model_dir),
        }
        with open('{}/train_config.json'.format(self.model_dir), 'w') as f:
            f.write(json.dumps(config, indent=4))

    def _transform_batch(self, batch_texts, batch_labels, max_length=512):
        """将batch的文本及labels转换为bert的输入tensor形式"""
        batch_input_ids, batch_att_mask, batch_label_ids = [], [], []
        for text, label in zip(batch_texts, batch_labels):  # 迭代每条样本
            assert isinstance(text, str)
            # 根据是否并行，获取bert_tokenizer
            bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel else self.model.module.bert_tokenizer
            encoded_dict = bert_tokenizer.encode_plus(text, max_length=max_length, padding='max_length',
                                                      return_tensors='pt', truncation=True)
            batch_input_ids.append(encoded_dict['input_ids'])
            batch_att_mask.append(encoded_dict['attention_mask'])
            batch_label_ids.append(self.vocab.tag2id[label])
        batch_input_ids, batch_att_mask, batch_label_ids = \
            torch.cat(batch_input_ids), torch.cat(batch_att_mask), torch.LongTensor(batch_label_ids)

        # 将数据拷贝到当前设备
        batch_input_ids, batch_att_mask, batch_label_ids = \
            batch_input_ids.to(self.device), batch_att_mask.to(self.device), batch_label_ids.to(self.device)

        return batch_input_ids, batch_att_mask, batch_label_ids

    def train(self, train_texts, train_labels, validate_texts, validate_labels,
              batch_size=30, epoch=10, max_loss_num=10, warning_max_len=True):
        """训练
        Args:
            train_texts: list[str] 训练集样本
            train_labels: list[str] 训练集标签
            validate_texts: list[str] 验证集样本
            validate_labels: list[str] 验证集标签
            batch_size: int
            epoch: int
            max_loss_num: int 连续max_loss_num个validate loss平均，根据其保存模型
            warning_max_len: bool 是否警告max_len超过512（普通bert不能处理超过512，除了一些变体如longformer）
        """
        self.batch_size = batch_size
        self.epoch = epoch

        # 构建词库表（只构建label id映射，词库bert已有）、构建模型
        self.vocab.build_vocab(labels=train_labels, build_texts=False, with_build_in_tag_id=False)
        self._build_model()
        # 保存词库、保存训练config
        self.vocab.save_vocab('{}/{}'.format(self.model_dir, self.vocab_name))
        self._save_config()

        logger.info('train samples: {}, validate samples: {}'.format(len(train_texts), len(validate_texts)))

        best_loss = float("inf")  # 当前最小的loss
        loss_buff = []  # 保存最近max_loss_num的个valid loss

        step = 0

        # 根据对抗配置，实例化不同的对抗实例
        if self.adversarial:
            logger.info('enable adversarial training')
            adv = FGM(self.model) if self.adversarial == 'fgm' else PGD(self.model)

        for epoch in range(epoch):
            for batch_idx in range(math.ceil(len(train_texts) / batch_size)):
                text_batch = train_texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
                labels_batch = train_labels[batch_size * batch_idx: batch_size * (batch_idx + 1)]

                step += 1
                self.model.train()  # 设置为train模式
                self.model.zero_grad()  # 清空梯度

                # 训练
                batch_max_len = max([len(text) for text in text_batch]) + 2  # 长度得加上[CLS]和[SEP]
                if warning_max_len and batch_max_len > 512:
                    logger.warning(
                        'current batch max_len is {}, > 512, which may not be processed by bert!'.format(batch_max_len)
                    )
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(text_batch,
                                                                                         labels_batch,
                                                                                         max_length=batch_max_len)
                best_paths, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
                if self.enable_parallel:  # 如果启用并行，需将多张卡返回的sub-batch loss平均
                    loss = loss.mean()
                loss.backward()

                # 对抗训练
                if self.adversarial:
                    adv.train(batch_input_ids, batch_att_mask, labels=batch_label_ids)

                # 更新参数
                self.optimizer.step()

                # 计算train acc、valid acc
                train_acc = self._get_acc_one_step(best_paths, batch_label_ids)
                valid_acc, valid_loss = self.validate(validate_texts, validate_labels, batch_size, warning_max_len)

                # 计算连续max_loss_num个平均valid_loss
                loss_buff.append(valid_loss)
                if len(loss_buff) > max_loss_num:
                    loss_buff = loss_buff[-max_loss_num:]
                avg_loss = sum(loss_buff) / len(loss_buff) if len(loss_buff) == max_loss_num else None

                logger.info(
                    'epoch %d, step %d, train loss %.4f, train acc %.4f, valid loss %.4f valid acc %.4f, '
                    'last %d avg valid loss %s' % (
                        epoch, step, loss, train_acc, valid_loss, valid_acc, max_loss_num,
                        '%.4f' % avg_loss if avg_loss else avg_loss
                    )
                )

                # 如果avg_loss在降，保存模型
                if avg_loss and avg_loss < best_loss:
                    best_loss = avg_loss
                    # 根据是否启用并行，获得state_dict
                    state_dict = self.model.state_dict() if not self.enable_parallel else self.model.module.state_dict()
                    torch.save(state_dict, '{}/{}'.format(self.model_dir, self.ckpt_name))
                    logger.info("model saved")

        logger.info("finished")

    def validate(self, validate_texts, validate_labels, sample_size=100, warning_max_len=True):
        """验证
        Args:
            validate_texts: list[str]
            validate_labels: list[str]
            sample_size: int 采样大小(使用全量验证集较慢，这里每次随机采样sample_size个样本做验证)
            warning_max_len: bool 是否警告max_len超过512（普通bert不能处理超过512，除了一些变体如longformer）
        Returns:
            float 验证集上acc, loss
        """
        # 设置为evaluate模式
        self.model.eval()

        # 随机采样sample_size个样本
        batch_texts, batch_labels = [
            return_val[:sample_size] for return_val in shuffle(validate_texts, validate_labels)
        ]

        # 计算valid acc, valid loss
        batch_max_len = max([len(text) for text in batch_texts]) + 2
        if warning_max_len and batch_max_len > 512:
            logger.warning(
                'current batch max_len is {}, > 512, which may not be processed by bert!'.format(batch_max_len)
            )
        with torch.no_grad():  # eval时不计算梯度
            batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(batch_texts,
                                                                                     batch_labels,
                                                                                     max_length=batch_max_len)
            best_paths, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
            if self.enable_parallel:
                loss = loss.mean()
            acc = self._get_acc_one_step(best_paths, batch_label_ids)
            return acc, loss

    def _get_acc_one_step(self, labels_predict_batch, labels_batch):
        """计算一个batch的acc, correct_label_num / total_label_num"""
        acc = (labels_predict_batch == labels_batch).sum().item() / labels_batch.shape[0]
        return float(acc)
