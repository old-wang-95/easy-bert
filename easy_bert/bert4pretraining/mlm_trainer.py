import json
import math
import os
import random
import re

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM

from easy_bert import logger


class MaskedLMTrainer(object):

    def __init__(self, pretrained_model_dir, model_dir, word_dict=frozenset(), learning_rate=5e-5,
                 ckpt_name='bert_model.bin', enable_parallel=False, random_seed=0, enable_fp16=False):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name

        self.learning_rate = learning_rate
        self.batch_size = None
        self.epoch = None

        self.enable_parallel = enable_parallel

        self.word_dict = word_dict
        if not word_dict:
            logger.warning('word_dict is empty, you can set it when enable whole word mask!')

        # 混合精度配置
        if enable_fp16:
            self.grad_scaler = GradScaler()  # 设置梯度缩放
        self.enable_fp16 = enable_fp16

        # 设置随机种子
        self.random_seed = random_seed
        self._set_random_seed(random_seed)

        # 自动获取当前设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_random_seed(self, seed):
        """针对torch torch.cuda numpy random分别设定随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self):
        """构建bert模型"""
        # 实例化BertForMaskedLM模型
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_dir)
        self.model = BertForMaskedLM.from_pretrained(self.pretrained_model_dir)

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

        # 启用并行，使用DataParallel封装model
        if self.enable_parallel:
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # 将模型拷贝到当前设备
        self.model.to(self.device)

    def _save_config(self):
        """保存训练参数配置为json文件"""
        config = {
            'vocab_size': self.bert_tokenizer.vocab_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'ckpt_name': self.ckpt_name,
            'enable_parallel': self.enable_parallel,
            'pretrained_model': os.path.basename(self.pretrained_model_dir),
        }
        with open('{}/train_config.json'.format(self.model_dir), 'w') as f:
            f.write(json.dumps(config, indent=4))

    def _get_word_token_positions(self, tokens, selected_position):
        """
        根据选中的token位置，获取全词对应的token位置
        tokenize结果和分词结果并不一致，这里为了简单，中文使用词库匹配，英文根据##标识
        """
        # 单字初始化
        word_token_ids, word = (selected_position,), tokens[selected_position]

        if re.match('[\u4e00-\u9fa5]', tokens[selected_position]):
            # 处理中文，词库匹配ngram，如果匹配到多个word，随机返回一个word
            words_candidate = set()
            for gram in [4, 3, 2]:
                for start in range(max(0, selected_position - gram + 1), selected_position + 1):
                    if len(tokens[start:start + gram]) == gram and \
                            ''.join(tokens[start:start + gram]) in self.word_dict:
                        words_candidate.add(((tuple(range(start, start + gram))), ''.join(tokens[start:start + gram])))
            if words_candidate:
                word_token_ids, word = random.sample(words_candidate, 1)[0]  # 随机采样一种切分
        elif re.search('[a-zA-Z]', tokens[selected_position]):
            # 处理英文，根据##判断是否要合并
            left, right = selected_position, selected_position + 1
            while left >= 0:
                if not tokens[left].startswith('##'):
                    break
                left -= 1
            while right < len(tokens):
                if not tokens[right].startswith('##'):
                    break
                right += 1
            word_token_ids, word = tuple(range(left, right)), ''.join(tokens[left:right])
        return word_token_ids, word

    def _transform_batch(self, batch_texts, max_length=512):
        """将batch的文本及labels转换为bert的输入tensor形式"""
        batch_input_ids, batch_att_mask, batch_label_ids = [], [], []
        for text in batch_texts:
            encoded_dict = self.bert_tokenizer.encode_plus(text, max_length=max_length, padding='max_length',
                                                           return_tensors='pt', truncation=True)
            input_ids, attention_mask = encoded_dict['input_ids'], encoded_dict['attention_mask']

            # 获取mask位置
            nopad_len = (input_ids != self.bert_tokenizer.pad_token_id).int().sum().item()  # 非pad部分长度
            tokens = self.bert_tokenizer.convert_ids_to_tokens([i.item() for i in input_ids[0][:nopad_len]])
            positions_valid = [  # 有效的positions
                i for i in range(1, nopad_len - 1) if tokens[i] != self.bert_tokenizer.unk_token
            ]
            positions_selected = sorted(  # 选15%的位置，不包括[CLS] [SEP] [PAD] [UNK]位置
                random.sample(positions_valid, round(0.15 * len(positions_valid)))
            )
            positions_mask = sorted(  # 80%真的mask
                random.sample(positions_selected, round(len(positions_selected) * 0.8))
            )
            positions_wwm_mask = set()
            for position in positions_mask:
                positions_wwm_mask |= set(self._get_word_token_positions(tokens, position)[0])  # 分词
            positions_wwm_mask = sorted(list(positions_wwm_mask))  # 全词mask
            positions_keep = sorted(  # 10%位置保留
                list(set(positions_selected) - set(positions_wwm_mask))[:round(len(positions_selected) * 0.1)]
            )
            positions_replace = sorted(  # 10%随机替换
                list(set(positions_selected) - set(positions_wwm_mask) - set(positions_keep))
            )

            # mask
            label_ids = input_ids.detach().clone()
            input_ids[0, positions_wwm_mask] = self.bert_tokenizer.mask_token_id  # 打mask
            for position_replace in positions_replace:  # 随机替换
                input_ids[0, position_replace] = random.randint(106, 8107)  # 随机采样一个词（中文或标点，参见vocab.txt）
            nomask_positions = \
                list(set(range(label_ids.size()[-1])) - set(positions_wwm_mask) - set(positions_keep) - set(
                    positions_replace))
            label_ids[0, nomask_positions] = -100  # 非mask的部分label全部设置为-100

            batch_input_ids.append(input_ids), batch_att_mask.append(attention_mask), batch_label_ids.append(label_ids)

        batch_input_ids, batch_att_mask, batch_label_ids = \
            torch.cat(batch_input_ids), torch.cat(batch_att_mask), torch.cat(batch_label_ids)

        # 将数据拷贝到当前设备
        batch_input_ids, batch_att_mask, batch_label_ids = \
            batch_input_ids.to(self.device), batch_att_mask.to(self.device), batch_label_ids.to(self.device)

        return batch_input_ids, batch_att_mask, batch_label_ids

    def train(self, train_texts, batch_size=30, epoch=10, warning_max_len=True):
        """训练
        Args:
            train_texts: list[str] 训练集样本
            batch_size: int
            epoch: int
            warning_max_len: bool 是否警告max_len超过512（普通bert不能处理超过512，除了一些变体如longformer）
        """
        self.batch_size = batch_size
        self.epoch = epoch

        self._build_model()
        self._save_config()

        logger.info('train samples: {}'.format(len(train_texts)))

        step = 0

        for epoch in range(epoch):
            for batch_idx in range(math.ceil(len(train_texts) / batch_size)):
                text_batch = train_texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]

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
                                                                                         max_length=batch_max_len)
                if self.enable_fp16:  # 如果启用混合精度训练，用autocast封装
                    with autocast():
                        loss, logits = self.model(
                            batch_input_ids, batch_att_mask, labels=batch_label_ids, return_dict=False
                        )
                else:  # 不启用混合精度，正常训练
                    loss, logits = self.model(
                        batch_input_ids, batch_att_mask, labels=batch_label_ids, return_dict=False
                    )

                labels_predict = torch.argmax(logits, dim=-1)
                train_acc = self._get_acc_one_step(labels_predict, batch_label_ids)

                # 如果启用并行，需将多张卡返回的sub-batch loss平均
                if self.enable_parallel:
                    loss = loss.mean()

                # 反向传播，并更新参数
                if self.enable_fp16:
                    self.grad_scaler.scale(loss).backward()  # 混合精度需要放大loss再反传
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                logger.info('epoch %d, step %d, train loss %.4f, train acc %.4f' % (epoch, step, loss, train_acc))

                # 1000个step存一次模型
                if step % 1000 == 0:
                    # 根据是否启用并行，获得state_dict
                    state_dict = self.model.state_dict() if not self.enable_parallel else self.model.module.state_dict()
                    torch.save(state_dict, '{}/{}'.format(self.model_dir, self.ckpt_name))
                    logger.info("model saved")

        logger.info("finished")

    def _get_acc_one_step(self, labels_predict_batch, labels_batch):
        """计算一个batch的所有label的acc, correct_label_num / total_label_num"""
        active_labels = labels_batch != -100
        total = labels_batch[active_labels].size()[-1]
        correct = (labels_batch[active_labels] == labels_predict_batch[active_labels]).int().sum().item()
        accuracy = correct / total
        return float(accuracy)
