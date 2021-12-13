import json

from easy_bert import logger


class Vocab(object):
    """
    词汇表

    基于texts构建vocab2id id2vocab，可以是char or word
    基于labels构建tag2id id2tag
    使用bert时，无需重新构建vocab2id id2vocab，直接set_vocab2id set_id2vocab即可
    """

    def __init__(self, vocab2id=None, id2vocab=None, tag2id=None, id2tag=None,
                 unk_vocab_id=0, pad_vocab_id=1, pad_tag_id=0):
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.vocab_size = 0 if not vocab2id else len(vocab2id)
        self.label_size = 0 if not tag2id else len(tag2id)
        self.unk_vocab_id = unk_vocab_id
        self.pad_vocab_id = pad_vocab_id
        self.pad_tag_id = pad_tag_id

    def build_vocab(self, texts=None, labels=None, build_texts=True, build_labels=True, with_build_in_tag_id=True):
        """构建词汇表"""
        logger.info('start to build vocab ...')

        if build_texts:
            assert texts, 'Please make sure texts is not None!'
            self.vocab2id, self.id2vocab = {'<UKN>': self.unk_vocab_id, '<PAD>': self.pad_vocab_id}, \
                                           {self.unk_vocab_id: '<UKN>', self.pad_vocab_id: '<PAD>'}
            vocab_cnt = 2
            for text in texts:
                for seg in text:
                    if seg in self.vocab2id:
                        continue
                    self.vocab2id[seg] = vocab_cnt
                    self.id2vocab[vocab_cnt] = seg
                    vocab_cnt += 1
            self.vocab_size = len(self.vocab2id)

        if build_labels:
            assert labels, 'Please make sure labels is not None!'
            self.tag2id, self.id2tag = {'<PAD>': self.pad_tag_id}, {self.pad_tag_id: '<PAD>'}
            tag_cnt = 1
            if not with_build_in_tag_id:  # label不预置PAD_ID
                self.tag2id, self.id2tag = {}, {}
                tag_cnt = 0
            for label in labels:
                if isinstance(label, list):
                    for each_label in label:
                        if each_label in self.tag2id:
                            continue
                        self.tag2id[each_label] = tag_cnt
                        self.id2tag[tag_cnt] = each_label
                        tag_cnt += 1
                else:  # label为str类型
                    assert isinstance(label, str)
                    self.tag2id[label] = tag_cnt
                    self.id2tag[tag_cnt] = label
                    tag_cnt += 1
            self.label_size = len(self.tag2id)

        logger.info('build vocab finish, vocab_size: {}, label_size: {}'.format(self.vocab_size, self.label_size))

    def save_vocab(self, vocab_file):
        """保存词汇表"""
        result = {
            'vocab2id': self.vocab2id,
            'id2vocab': self.id2vocab,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
        logger.info('save vocab to {}'.format(vocab_file))

    def load_vocab(self, vocab_file):
        """加载词汇表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            result = json.loads(f.read())
        self.vocab2id = result['vocab2id']
        self.id2vocab = {int(k): v for k, v in result['id2vocab'].items()}  # 将id转为int
        self.tag2id = result['tag2id']
        self.id2tag = {int(k): v for k, v in result['id2tag'].items()}
        self.vocab_size = len(self.vocab2id)
        self.label_size = len(self.tag2id)

    def set_vocab2id(self, vocab2id):
        self.vocab2id = vocab2id
        self.vocab_size = len(self.vocab2id)
        return self

    def set_id2vocab(self, id2vocab):
        self.id2vocab = id2vocab
        return self

    def set_tag2id(self, tag2id):
        self.tag2id = tag2id
        self.label_size = len(self.tag2id)
        return self

    def set_id2tag(self, id2tag):
        self.id2tag = id2tag
        return self

    def set_unk_vocab_id(self, unk_vocab_id):
        self.unk_vocab_id = unk_vocab_id

    def set_pad_vocab_id(self, pad_vocab_id):
        self.pad_vocab_id = pad_vocab_id

    def set_pad_tag_id(self, pad_tag_id):
        self.pad_tag_id = pad_tag_id
