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
            img_dim=cfg.swin_transunet.img_dim,
            in_channels=cfg.swin_transunet.in_channels,
            out_channels=cfg.swin_transunet.out_channels,
            depths=cfg.swin_transunet.depths,
            num_heads=cfg.swin_transunet.num_heads,
            window_size=cfg.swin_transunet.window_size,
            class_num=cfg.swin_transunet.class_num
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
