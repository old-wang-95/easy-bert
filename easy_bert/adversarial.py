import torch

"""
对抗训练

在embedding输出上加扰动，即在embedding层权重上加扰动
embedding层即一个特殊的全连接层，输入onehot查询向量 seq_len * vocab_size，权重 vocab_size * embedding_size
"""


class FGM(object):
    """
    Fast Gradient Method（FGM）
    """

    def __init__(self, model, epsilon=1., emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name

        self.backup = {}

    def attack(self):
        """对抗，计算embedding层扰动r，并相加"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:  # 找到embedding层参数
                self.backup[name] = param.data.clone()  # 备份参数到backup
                norm = torch.norm(param.grad)  # 计算2范数
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm  # 计算fgm扰动r
                    param.data.add_(r_adv)  # x + r

    def restore(self):
        """恢复embedding层参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]  # 从backup中还原原始embedding层参数
        self.backup = {}  # 清空backup

    def train(self, *args, **kwargs):
        """对抗训练"""
        self.attack()  # 在embedding上增加扰动
        _, loss_adv = self.model(*args, **kwargs)  # 使用扰动后的embedding计算新loss
        loss_adv.backward()  # 反向传播，在正常梯度上，累加扰动后的梯度
        self.restore()  # 恢复embedding层参数


class PGD(object):
    """
    Projected Gradient Descent（PGD）
    """

    def __init__(self, model, epsilon=1., alpha=0.3, emb_name='word_embeddings', k=3):
        self.model = model  # 对抗的model
        self.epsilon = epsilon  # 最大扰动半径
        self.alpha = alpha  # 步长
        self.emb_name = emb_name
        self.k = k  # 对抗阶数

        self.emb_backup = {}  # 备份embedding参数
        self.grad_backup = {}  # 备份梯度

    def attack(self, is_first_attack=False):
        """对抗，计算embedding层扰动r，并相加"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:  # 找到embedding层参数
                if is_first_attack:  # 第一次对抗时，备份embedding参数
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 计算2范数
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.alpha * param.grad / norm  # 计算扰动r
                    param.data.add_(r_adv)  # x + r
                    param.data = self.project(name, param.data, self.epsilon)  # 扰动半径超过epsilon，投影回来

    def restore(self):
        """恢复embedding层参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        """投影，确保扰动r不超过扰动半径"""
        r = param_data - self.emb_backup[param_name]  # 获得当前扰动半径r
        if torch.norm(r) > epsilon:  # 如果扰动半径过大，投影（二范数归一）
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r  # 返回新的 x + r

    def backup_grad(self):
        """备份对抗前所有参数梯度"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """恢复对抗前所有参数梯度"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

    def train(self, *args, **kwargs):
        """对抗训练"""
        # 备份梯度
        self.backup_grad()

        # 进行k阶对抗
        # 前k-1次model call主要计算扰动r
        # 最后1次model call使用扰动后的embedding计算对抗loss，累加梯度
        for k_i in range(self.k):
            # 对抗，对embedding添加扰动，（第1次对抗时备份embedding参数）
            self.attack(is_first_attack=(k_i == 0))
            if k_i != self.k - 1:
                self.model.zero_grad()  # 前几次对抗时，清空梯度，便于计算k_i时刻新梯度
            else:
                self.restore_grad()  # 最后一次对抗时，恢复对抗前所有参数梯度
            _, loss_adv = self.model(*args, **kwargs)  # 使用扰动后的embedding计算新loss
            loss_adv.backward()  # 反向传播，计算新梯度（最后一次时，累加扰动的梯度）

        self.restore()  # 恢复embedding层参数
