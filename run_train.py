from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

# 导入项目模块
from utils import transforms as T
from utils.dataset import DentalDataset
from utils.utils import EpochCallback
from config import cfg
from train_swin_transunet import SwinTransUNetSeg


class TrainPipeline:
    def __init__(self, train_path, test_path, model_path, device):
        self.device = device
        self.model_path = model_path

        # 加载数据集
        self.train_loader = self.__load_dataset(train_path, train=True)
        self.test_loader = self.__load_dataset(test_path)

        # 初始化Swin TransUNet模型
        self.swin_transunet = SwinTransUNetSeg(self.device)

    def __load_dataset(self, path, train=False):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            transform = transforms.Compose([T.RandomAugmentation(2)])

        dataset = DentalDataset(path, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0

        for step, data in enumerate(loader):
            img, mask = data['img'], data['mask']
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_func(img=img, mask=mask)

            total_loss += loss

            t.update()

        return total_loss

    def train(self):
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.swin_transunet.model, self.swin_transunet.optimizer, 'test_loss', cfg.patience)

        print(f"开始训练 Swin TransUNet 模型，总共 {cfg.epoch} 个轮次")
        print(f"训练数据: {len(self.train_loader)} 批次, 测试数据: {len(self.test_loader)} 批次")
        
        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                t.set_description(f"轮次 {epoch+1}/{cfg.epoch}")
                
                # 训练阶段
                train_loss = self.__loop(self.train_loader, self.swin_transunet.train_step, t)
                
                # 测试阶段
                test_loss = self.__loop(self.test_loader, self.swin_transunet.test_step, t)

            # 计算平均损失并更新回调
            avg_train_loss = train_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)
            
            print(f"轮次 {epoch+1}/{cfg.epoch} - 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}")
            
            callback.epoch_end(epoch + 1,
                               {'loss': avg_train_loss,
                                'test_loss': avg_test_loss})

            if callback.end_training:
                print("提前停止训练，达到最佳模型")
                break

        print(f"训练完成！最佳模型已保存到 {self.model_path}")


def main():
    parser = argparse.ArgumentParser(description="Swin TransUNet 训练脚本")
    parser.add_argument('--train_path', type=str, default="./DRIVE/training", help='训练数据集路径')
    parser.add_argument('--test_path', type=str, default="./DRIVE/test", help='测试数据集路径')
    parser.add_argument('--model_path', type=str, default="./model_swin_transunet.pth", help='模型保存路径')
    args = parser.parse_args()

    # 检测设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行训练")

    # 初始化训练管道并开始训练
    trainer = TrainPipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        device=device
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
