import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import LSTM, GRU
from transformers import AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import LongformerModel

from easy_bert import logger
from easy_bert.losses.crf_layer import CRF
from easy_bert.losses.focal_loss import FocalLoss
from easy_bert.losses.label_smoothing_loss import LabelSmoothingCrossEntropy
from easy_bert.modeling_nezha import NeZhaModel


class SequenceLabelingModel(nn.Module):

    def __init__(
            self,
            bert_base_model_dir, label_size, drop_out_rate=0.5,
            loss_type='crf_loss', focal_loss_gamma=2, focal_loss_alpha=None,
            add_on=None, rnn_hidden=256
    ):
        super(SequenceLabelingModel, self).__init__()
        self.label_size = label_size

        assert loss_type in ('crf_loss', 'cross_entropy_loss', 'focal_loss', 'label_smoothing_loss')
        if focal_loss_alpha:  # 确保focal_loss_alpha合法，必须是一个label的概率分布
            assert isinstance(focal_loss_alpha, list) and len(focal_loss_alpha) == label_size
        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        # bert附加层，可以不接或者接BiLSTM或BiGRU
        assert add_on in (None, 'bilstm', 'bigru')
        self.add_on = add_on
        self.rnn_hidden = rnn_hidden

        # 自动获取当前设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据预训练文件名，自动检测bert的各种变体，并加载
        if 'albert' in bert_base_model_dir.lower():
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = AlbertModel.from_pretrained(bert_base_model_dir)
        elif 'electra' in bert_base_model_dir.lower():
            self.bert_tokenizer = ElectraTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = ElectraModel.from_pretrained(bert_base_model_dir)
        elif 'longformer' in bert_base_model_dir.lower():
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            # longformer-chinese-base-4096模型参数prefix为bert而非标准的longformer，这是个坑
            LongformerModel.base_model_prefix = 'bert'
            self.bert_model = LongformerModel.from_pretrained(bert_base_model_dir)
        elif 'distil' in bert_base_model_dir.lower():
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = DistilBertModel.from_pretrained(bert_base_model_dir)
        elif 'nezha' in bert_base_model_dir.lower():
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = NeZhaModel.from_pretrained(
                bert_base_model_dir, output_hidden_states=True, output_attentions=True
            )
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = BertModel.from_pretrained(bert_base_model_dir)

        logger.info('tokenizer: {}, bert_model: {}'.
                    format(self.bert_tokenizer.__class__.__name__, self.bert_model.__class__.__name__))

        self.dropout = nn.Dropout(drop_out_rate)

        linear_layer_input_size = self.bert_model.config.hidden_size  # 分类层输入size
        # 附加层
        if self.add_on:
            rnn_class = LSTM if self.add_on == 'bilstm' else GRU
            self.rnn = rnn_class(
                self.bert_model.config.hidden_size, self.rnn_hidden, batch_first=True, bidirectional=True
            )
            linear_layer_input_size = 2 * self.rnn_hidden

        # 定义linear层，将hidden_size映射到label_size层
        self.linear = nn.Linear(linear_layer_input_size, label_size)

        # 定义crf层
        self.crf = CRF(label_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, return_extra=False):

        if isinstance(self.bert_model, LongformerModel):
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, inputs_embeds=None, return_dict=False,
                output_hidden_states=True,  # longformer直接output_attentions会报错，这里暂时将其attentions置为None
            ) + (None,)
        elif isinstance(self.bert_model, DistilBertModel):
            # distilbert不支持token_type_ids、position_ids参数，不传入
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=None, return_dict=False,
                output_hidden_states=True, output_attentions=True
            )
        elif isinstance(self.bert_model, NeZhaModel):
            # nazhe模型会少一些参数
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, inputs_embeds=None,
            )
        else:
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, inputs_embeds=None, return_dict=False,
                output_hidden_states=True, output_attentions=True
            )
        bert_out, (hidden_states, attentions) = bert_out[:-2], bert_out[-2:]

        last_hidden_state = bert_out[0]

        linear_layer_input = last_hidden_state
        if self.add_on:
            rnn_out, _ = self.rnn(last_hidden_state)  # (batch,seq,bert_hidden) -> (batch,seq,2*rnn_hidden)
            linear_layer_input = rnn_out

        logits = self.linear(self.dropout(linear_layer_input))

        # 根据loss_type，选择使用维特比解码或直接argmax
        if self.loss_type == 'crf_loss':
            best_paths, scores = self.crf.viterbi_decode(logits, attention_mask)
        else:
            best_paths = torch.argmax(logits, dim=-1)

        # 将logits hiddens attentions装进extra，蒸馏时可能需要使用
        extra = {'hiddens': hidden_states, 'logits': logits, 'attentions': attentions}

        if labels is not None:
            # 根据不同的loss_type，选择不同的loss计算
            active_loss = attention_mask.view(-1) == 1  # 通过attention_mask忽略pad
            active_logits, active_labels = logits.view(-1, self.label_size)[active_loss], labels.view(-1)[active_loss]
            if self.loss_type == 'crf_loss':
                # 计算loss时，忽略[CLS]、[SEP]以及PAD部分
                lengths, new_att_mask = attention_mask.sum(axis=1), attention_mask.clone()
                for i, length in enumerate(lengths):
                    new_att_mask[i, length - 1] = 0  # [SEP]部分置为0
                loss = self.crf.forward(logits[:, 1:, :], labels[:, 1:], new_att_mask[:, 1:]).mean()
            elif self.loss_type == 'cross_entropy_loss':
                loss = CrossEntropyLoss(ignore_index=-1)(active_logits, active_labels)  # 忽略label=-1位置
            elif self.loss_type == 'focal_loss':
                active_loss = active_labels != -1  # 忽略label=-1位置，即[CLS]和[SEP]
                active_logits, active_labels = active_logits[active_loss], active_labels[active_loss]
                loss = FocalLoss(gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha)(active_logits, active_labels)
            else:
                loss = LabelSmoothingCrossEntropy(alpha=0.1, ignore_index=-1)(active_logits, active_labels)
            return (best_paths, loss) if not return_extra else (best_paths, loss, extra)

        return best_paths if not return_extra else (best_paths, extra)

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
