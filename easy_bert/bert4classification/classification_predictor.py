import json
import math

import torch
import torch.onnx
from onnxruntime import InferenceSession
from torch.nn import DataParallel

from easy_bert.base.base_predictor import BasePredictor
from easy_bert.bert4classification.classification_model import ClassificationModel
from easy_bert.vocab import Vocab


class ClassificationPredictor(BasePredictor):
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

        # onnx相关配置
        self.is_onnx_model = False  # 当前模型是否为onnx模型
        self.onnx_model = None  # onnx_model初始化为None

    def _load_config(self):
        """json加载训练config"""
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        """加载模型"""
        # 根据config初始化ClassificationModel
        # 注意，部分参数推理时并不需要，这里传入debug时可读性更好
        self.model = ClassificationModel(
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
            texts: list[str] 预测样本
            batch_size: int
            max_len: int 最大序列长度
        Returns:
            list[str] 标签序列
        """
        batch_labels = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]

            # 当前batch最大长度
            batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)

            # 将texts转化为tensor
            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                assert isinstance(text, str)
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
            with torch.no_grad():  # 推理时不计算梯度
                if self.is_onnx_model:  # onnx model推理
                    best_labels = self.onnx_model.run(
                        ['output'],  # 设置输出names
                        # feed输入，并转化为numpy数组
                        {'input1': batch_input_ids.cpu().numpy(), 'input2': batch_att_mask.cpu().numpy()}
                    )[0]
                else:  # torch模型推理
                    best_labels = self.model(batch_input_ids, batch_att_mask)
                batch_labels.extend([self.vocab.id2tag[label_id.item()] for label_id in best_labels])

        return batch_labels

    def get_bert_tokenizer(self):
        """根据是否并行，获取bert_tokenizer"""
        bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel else self.model.module.bert_tokenizer
        return bert_tokenizer

    def transform2onnx(self):
        """将模型转换为onnx模型"""
        assert not self.is_onnx_model, 'error, curl model is already onnx model!'

        # 定义伪输入，让onnx做一遍推理，构建静态计算图
        dummy_inputs = torch.LongTensor([[i for i in range(200)]])
        dummy_att_masks = torch.LongTensor([[1 for _ in range(200)]])
        dummy_inputs, dummy_att_masks = dummy_inputs.to(self.device), dummy_att_masks.to(self.device)

        # 将模型导出为onnx标准
        torch.onnx.export(
            self.model, (dummy_inputs, dummy_att_masks),
            '{}/model.onnx'.format(self.model_dir),
            # 设置model的输入输出，参考ClassificationModel.forward函数签名
            input_names=['input1', 'input2'], output_names=['output'],  # 两个输入，一个输出
            # 设置batch、seq_len维度可变
            dynamic_axes={'input1': {0: 'batch', 1: 'seq'}, 'input2': {0: 'batch', 1: 'seq'}, 'output': {0: 'batch'}},
            opset_version=10,
        )

        # 加载onnx model
        self._load_onnx_model()

    def _load_onnx_model(self):
        """加载onnx model"""
        # 通过InferenceSession，加载onnx模型
        self.onnx_model = InferenceSession('{}/model.onnx'.format(self.model_dir))
        self.onnx_model.get_modelmeta()

        # 更新is_cur_onnx_model状态
        self.is_onnx_model = True
