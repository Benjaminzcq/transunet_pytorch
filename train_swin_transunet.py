import torch
from torch.optim import SGD

# Additional Scripts
from utils.swin_transunet import SwinTransUNet
from utils.utils import dice_loss
from config import cfg


class SwinTransUNetSeg:
    def __init__(self, device):
        self.device = device
        self.model = SwinTransUNet(
            img_dim=cfg.transunet.img_dim,
            in_channels=cfg.transunet.in_channels,
            out_channels=64,  # 减小输出通道数以减少显存消耗
            depths=[2, 2, 4, 2],  # 减少Swin Transformer的深度
            num_heads=[2, 4, 8, 16],  # 减少注意力头数
            window_size=7,  # 窗口大小
            class_num=cfg.transunet.class_num
        ).to(self.device)

        self.criterion = dice_loss
        self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate,
                             momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred_mask

    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])

        return loss.item(), pred_mask
