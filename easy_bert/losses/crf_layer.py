import torch
import torch.nn as nn

"""
本代码主要来自fastNLP
https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/decoder/crf.py
"""


class CRF(nn.Module):
    r"""
    条件随机场。提供forward()以及viterbi_decode()两个方法，分别用于训练与inference。
    """

    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None):
        r"""

        :param int num_tags: 标签的数量
        :param bool include_start_end_trans: 是否考虑各个tag作为开始以及结尾的分数。
        :param List[Tuple[from_tag_id(int), to_tag_id(int)]] allowed_transitions: 内部的Tuple[from_tag_id(int),
                                   to_tag_id(int)]视为允许发生的跃迁，其他没有包含的跃迁认为是禁止跃迁，可以通过
                                   allowed_transitions()函数得到；如果为None，则所有跃迁均为合法
        """
        super(CRF, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            has_start = False
            has_end = False
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
                if from_tag_id == num_tags:
                    has_start = True
                if to_tag_id == num_tags + 1:
                    has_end = True
            if not has_start:
                constrain[num_tags, :].fill_(0)
            if not has_end:
                constrain[:, num_tags + 1].fill_(0)
        self._constrain = nn.Parameter(constrain, requires_grad=False)

        nn.init.xavier_normal_(self.trans_m)

    def _normalizer_likelihood(self, logits, mask):
        r"""Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        :param logits:FloatTensor, max_len x batch_size x num_tags
        :param mask:ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha = alpha + self.start_scores.view(1, -1)

        flip_mask = mask.eq(False)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                    alpha.masked_fill(mask[i].eq(True).view(batch_size, 1), 0)

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(self, logits, tags, mask):
        r"""
        Compute the score for the gold path.
        :param logits: FloatTensor, max_len x batch_size x num_tags
        :param tags: LongTensor, max_len x batch_size
        :param mask: ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        mask = mask.eq(True)
        flip_mask = mask.eq(False)
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score = score + st_scores + ed_scores
        # return [B,]
        return score

    def forward(self, feats, tags, mask):
        r"""
        用于计算CRF的前向loss，返回值为一个batch_size的FloatTensor，可能需要mean()求得loss。
        :param torch.FloatTensor feats: batch_size x max_len x num_tags，特征矩阵。
        :param torch.LongTensor tags: batch_size x max_len，标签矩阵。
        :param torch.ByteTensor mask: batch_size x max_len，为0的位置认为是padding。
        :return: torch.FloatTensor, (batch_size,)
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)

        return all_path_score - gold_path_score

    def viterbi_decode(self, logits, mask, unpad=False):
        r"""给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数
        :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
        :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
        :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
            List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
            个sample的有效长度。
        :return: 返回 (paths, scores)。
                    paths: 是解码后的路径, 其值参照unpad参数.
                    scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。
        """
        batch_size, max_len, n_tags = logits.size()
        seq_len = mask.long().sum(1)
        logits = logits.transpose(0, 1).data  # L, B, H
        mask = mask.transpose(0, 1).data.eq(True)  # L, B
        flip_mask = mask.eq(False)

        # dp
        vpath = logits.new_zeros((max_len, batch_size, n_tags), dtype=torch.long)
        vscore = logits[0]  # bsz x n_tags
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags + 1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]

        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        end_trans_score = transitions[:n_tags, n_tags + 1].view(1, 1, n_tags).repeat(batch_size, 1, 1)  # bsz, 1, n_tags

        # 针对长度为1的句子
        vscore += transitions[:n_tags, n_tags + 1].view(1, n_tags).repeat(batch_size, 1) \
            .masked_fill(seq_len.ne(1).view(-1, 1), 0)
        for i in range(1, max_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = logits[i].view(batch_size, 1, n_tags) + trans_score
            score = prev_score + cur_score.masked_fill(flip_mask[i].view(batch_size, 1, 1), 0)  # bsz x n_tag x n_tag
            # 需要考虑当前位置是该序列的最后一个
            score += end_trans_score.masked_fill(seq_len.ne(i + 1).view(-1, 1, 1), 0)

            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            # 由于最终是通过last_tags回溯，需要保持每个位置的vscore情况
            vscore = best_score.masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                     vscore.masked_fill(mask[i].view(batch_size, 1), 0)

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(max_len, dtype=torch.long, device=logits.device)
        lens = (seq_len - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % max_len

        ans = logits.new_empty((max_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(max_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, max_len in enumerate(lens):
                paths.append(ans[idx, :max_len + 1].tolist())
        else:
            paths = ans
        return paths, ans_score
