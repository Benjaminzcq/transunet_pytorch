import os
import cv2
import torch
import numpy as np
import datetime
import argparse

# Additional Scripts
from train_swin_transunet import SwinTransUNetSeg
from utils.utils import thresh_func
from config import cfg


class SwinSegInference:
    def __init__(self, model_path, device):
        self.device = device
        self.swin_transunet = SwinTransUNetSeg(device)
        self.swin_transunet.load_model(model_path)

        if not os.path.exists('./results'):
            os.mkdir('./results')

    def read_and_preprocess(self, p):
        # 检查掩码文件扩展名
        if p.lower().endswith('.gif'):
            # 使用PIL读取GIF文件
            from PIL import Image
            img_pil = Image.open(p)
            img = np.array(img_pil.convert('RGB'))
        else:
            # 使用OpenCV读取其他格式
            img = cv2.imread(p)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_torch = cv2.resize(img, (cfg.transunet.img_dim, cfg.transunet.img_dim))
        img_torch = img_torch / 255.
        img_torch = img_torch.transpose((2, 0, 1))
        img_torch = np.expand_dims(img_torch, axis=0)
        img_torch = torch.from_numpy(img_torch.astype('float32')).to(self.device)

        return img, img_torch

    def save_preds(self, preds, folder_name=None):
        if folder_name is None:
            folder_path = './results/' + str(datetime.datetime.utcnow()).replace(' ', '_')
        else:
            folder_path = './results/' + folder_name

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for name, pred_mask in preds.items():
            cv2.imwrite(f'{folder_path}/{name}', pred_mask)
            print(f'保存预测结果到: {folder_path}/{name}')

    def infer(self, path, merged=True, save=True, folder_name=None):
        path = [path] if isinstance(path, str) else path

        preds = {}
        for p in path:
            file_name = p.split('/')[-1]
            print(f'处理图像: {file_name}')
            img, img_torch = self.read_and_preprocess(p)
            with torch.no_grad():
                pred_mask = self.swin_transunet.model(img_torch)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))

            orig_h, orig_w = img.shape[:2]
            pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
            pred_mask = thresh_func(pred_mask, thresh=cfg.inference_threshold)
            pred_mask *= 255

            if merged:
                pred_mask = cv2.bitwise_and(img, img, mask=pred_mask.astype('uint8'))

            preds[file_name] = pred_mask

        if save:
            self.save_preds(preds, folder_name)

        return preds


def main():
    parser = argparse.ArgumentParser(description='Swin TransUNet 推理')
    parser.add_argument('--model_path', type=str, default="./model_swin_transunet.pth", help='模型权重路径')
    parser.add_argument('--image_path', type=str, default="./DRIVE/test/images", help='输入图像路径，可以是单张图像或包含多张图像的文件夹')
    parser.add_argument('--output_folder', type=str, default="swin_transunet_results", help='输出文件夹名称，默认使用时间戳')
    parser.add_argument('--merged', action='store_true', default=True, help='是否将分割结果与原图合并')
    parser.add_argument('--no_save', action='store_true', default=False, help='是否不保存结果')
    args = parser.parse_args()

    # 检测GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 初始化推理类
    inference = SwinSegInference(args.model_path, device)

    # 处理输入路径
    if os.path.isdir(args.image_path):
        # 如果是文件夹，获取所有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.gif']:
            image_files.extend(glob.glob(os.path.join(args.image_path, ext)))
        if not image_files:
            print(f'在 {args.image_path} 中未找到图像文件')
            return
        print(f'找到 {len(image_files)} 个图像文件')
    else:
        # 单个文件
        if not os.path.exists(args.image_path):
            print(f'图像文件 {args.image_path} 不存在')
            return
        image_files = [args.image_path]

    # 执行推理
    results = inference.infer(
        path=image_files,
        merged=args.merged,
        save=not args.no_save,
        folder_name=args.output_folder
    )

    print('推理完成!')


if __name__ == '__main__':
    import glob
    main()
