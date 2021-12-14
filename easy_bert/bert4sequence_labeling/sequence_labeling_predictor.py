import json
import math

import torch
from torch.nn import DataParallel

from easy_bert.base.base_predictor import BasePredictor
from easy_bert.bert4sequence_labeling.sequence_labeling_model import SequenceLabelingModel
from easy_bert.vocab import Vocab


class SequenceLabelingPredictor(BasePredictor):
    def __init__(self, pretrained_model_dir, model_dir, vocab_name='vocab.json',
                 enable_parallel=False):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.enable_parallel = enable_parallel

        # 自动获取当前设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab = Vocab()

        # 加载config、vocab、model
        self._load_config()
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()

    def _load_config(self):
        """json加载训练config"""
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        """加载模型"""
        # 根据config初始化SequenceLabelingModel
        # 注意，部分参数推理时并不需要，这里传入debug时可读性更好
        self.model = SequenceLabelingModel(
            self.pretrained_model_dir, self._config['label_size'], self._config['dropout_rate'],
            loss_type=self._config['loss_type'], focal_loss_alpha=self._config['focal_loss_alpha'],
            focal_loss_gamma=self._config['focal_loss_gamma']
        )

        # 加载模型文件里的参数
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir, self._config['ckpt_name']), map_location=self.device)
        )

        # 设置为evaluate模式
        self.model.eval()

        # 如果启用并行，使用DataParallel封装model
        if self.enable_parallel:
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # 将模型拷贝到当前设备
        self.model.to(self.device)

        # 手动设置vocab中unk、pad标签id
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])

    def predict(self, texts, batch_size=64, max_len=512):
        """
        Args:
            texts: list[list[str]] 预测样本
            batch_size: int
            max_len: int 最大序列长度
        Returns:
            list[list[str]] 标签序列
        """
        batch_labels = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]

            # 当前batch最大长度
            batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)

            # 将texts转化为tensor
            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                assert isinstance(text, list)
                text = ' '.join(text)  # 确保输入encode_plus函数为文本
                bert_tokenizer = self.get_bert_tokenizer()
                encoded_dict = bert_tokenizer.encode_plus(text, max_length=batch_max_len,
                                                          padding='max_length',
                                                          return_tensors='pt', truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'])
                batch_att_mask.append(encoded_dict['attention_mask'])
            batch_input_ids, batch_att_mask = torch.cat(batch_input_ids), torch.cat(batch_att_mask)

            # 将数据拷贝到当前设备
            batch_input_ids, batch_att_mask = batch_input_ids.to(self.device), batch_att_mask.to(self.device)

            # 推理，并将结果解析为原始label
            with torch.no_grad():
                best_paths = self.model(batch_input_ids, batch_att_mask)
                for best_path, att_mask in zip(best_paths, batch_att_mask):
                    active_labels = best_path[att_mask == 1][1:-1]  # 截掉pad、[CLS]、[SEP]部分
                    labels = [self.vocab.id2tag[label_id.item()] for label_id in active_labels]
                    batch_labels.append(labels)

        return batch_labels

    def get_bert_tokenizer(self):
        """根据是否并行，获取bert_tokenizer"""
        bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel else self.model.module.bert_tokenizer
        return bert_tokenizer
