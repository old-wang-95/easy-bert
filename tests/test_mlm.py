import unittest

from easy_bert.bert4pretraining.mlm_trainer import MaskedLMTrainer


class MyTestCase(unittest.TestCase):

    def test(self):
        print('test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        texts = [
            '早上起床后，我发现今天天气还真是不错的。早上起床后，我发现今天天气还真是不错的。早上起床后，我发现今天天气还真是不错的。'
        ]

        word_dict = {'今天', '天气', '早上', '起床', '发现', '不错'}  # 词库，用来全词mask
        pretrained_model_dir, model_dir = './models/chinese-roberta-wwm-ext', './tests/test_model'
        trainer = MaskedLMTrainer(
            pretrained_model_dir, model_dir, word_dict=word_dict,
            learning_rate=5e-5, enable_parallel=False, random_seed=0, enable_fp16=False
        )

        trainer.train(texts, batch_size=1, epoch=10)


if __name__ == '__main__':
    unittest.main()
