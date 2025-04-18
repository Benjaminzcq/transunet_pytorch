import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Additional Scripts
from config import cfg


class DentalDataset(Dataset):
    output_size = cfg.transunet.img_dim

    def __init__(self, path, transform):
        super().__init__()

        self.transform = transform

        img_folder = os.path.join(path, 'images')
        mask_folder = os.path.join(path, '1st_manual')  # 修改为使用1st_manual文件夹

        # 确保文件夹存在
        if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
            raise ValueError(f"数据集路径不正确，请检查：{path}")

        # 获取所有图像文件
        img_files = sorted(os.listdir(img_folder))
        mask_files = sorted(os.listdir(mask_folder))

        # 打印调试信息
        print(f"找到 {len(img_files)} 个图像文件和 {len(mask_files)} 个掩码文件")
        if len(img_files) > 0 and len(mask_files) > 0:
            print(f"图像示例: {img_files[0]}, 掩码示例: {mask_files[0]}")

        self.img_paths = []
        self.mask_paths = []

        # 为DRIVE数据集处理
        # 图像格式: xx_training.png 或 xx_test.png
        # 掩码格式: xx_training_mask.png 或 xx_test_mask.png
        for img_file in img_files:
            # 跳过非PNG文件
            if not img_file.lower().endswith('.png'):
                continue
                
            img_id = img_file.split('_')[0]  # 提取ID号
            img_path = os.path.join(img_folder, img_file)
            
            # 查找对应的掩码文件
            matching_mask = None
            for mask_file in mask_files:
                if mask_file.startswith(img_id + '_') and mask_file.lower().endswith('.png'):
                    matching_mask = os.path.join(mask_folder, mask_file)
                    break
            
            if matching_mask is not None:
                self.img_paths.append(img_path)
                self.mask_paths.append(matching_mask)
            else:
                print(f"警告: 未找到图像 {img_file} 对应的掩码文件")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 使用OpenCV读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        img = cv2.resize(img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        
        # 使用OpenCV读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩码（确保只有0和255）
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # 添加通道维度
        mask = np.expand_dims(mask, axis=-1)

        sample = {'img': img, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        img, mask = sample['img'], sample['mask']

        # 将NumPy数组转换为PyTorch张量
        try:
            # 强制转换为标准NumPy数组
            if isinstance(img, np.ndarray) and img.dtype == np.object_:
                # 如果是object类型，尝试将其转换为标准数组
                img_list = []
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        img_list.append(img[i, j])
                img = np.array(img_list).reshape(img.shape[0], img.shape[1], -1)
            
            # 归一化图像
            img = np.array(img, dtype=np.float32) / 255.0
            
            # 确保图像是3通道
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            img = img.transpose((2, 0, 1))  # 调整通道顺序为(C,H,W)
            img = torch.FloatTensor(img)
            
            # 处理掩码
            if isinstance(mask, np.ndarray) and mask.dtype == np.object_:
                # 如果是object类型，尝试将其转换为标准数组
                mask_list = []
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        mask_list.append(mask[i, j])
                mask = np.array(mask_list).reshape(mask.shape[0], mask.shape[1], -1)
            
            # 归一化掩码
            mask = np.array(mask, dtype=np.float32) / 255.0
            
            # 确保掩码是单通道或三通道
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)
                
            mask = mask.transpose((2, 0, 1))
            mask = torch.FloatTensor(mask)
            
        except Exception as e:
            print(f"数据转换错误: {e}")
            print(f"图像类型: {type(img)}, 掩码类型: {type(mask)}")
            if isinstance(img, np.ndarray):
                print(f"图像形状: {img.shape}, 数据类型: {img.dtype}")
            if isinstance(mask, np.ndarray):
                print(f"掩码形状: {mask.shape}, 数据类型: {mask.dtype}")
            raise

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from utils import transforms as T

    transform = transforms.Compose([T.BGR2RGB(),
                                    T.Rescale(cfg.input_size),
                                    T.RandomAugmentation(2),
                                    T.Normalize(),
                                    T.ToTensor()])

    md = DentalDataset('/home/kara/Downloads/UFBA_UESC_DENTAL_IMAGES_DEEP/dataset_and_code/test/set/train',
                       transform)

    for sample in md:
        print(sample['img'].shape)
        print(sample['mask'].shape)
        '''cv2.imshow('img', sample['img'])
        cv2.imshow('mask', sample['mask'])
        cv2.waitKey()'''

        break
