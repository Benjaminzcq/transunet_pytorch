import os
import cv2
import torch
import numpy as np
import datetime
import argparse
import traceback
from PIL import Image

# Additional Scripts
from model_swin_transunet import SwinTransUNetSeg
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
        # 使用PIL读取图像（支持所有格式）
        pil_img = Image.open(p)
        pil_img_rgb = pil_img.convert('RGB')
        img = np.array(pil_img_rgb)
        
        # 调整图像大小为模型输入尺寸
        pil_img_resized = pil_img_rgb.resize((cfg.transunet.img_dim, cfg.transunet.img_dim), Image.BILINEAR)
        img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0  # 归一化
        
        # 转换为PyTorch张量
        img_np_array = np.asarray(img_np, dtype=np.float32)
        img_torch = torch.tensor(img_np_array.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)

        return img, img_torch

    def save_preds(self, preds, folder_name=None):
        if folder_name is None:
            folder_path = './results/' + str(datetime.datetime.utcnow()).replace(' ', '_')
        else:
            folder_path = './results/' + folder_name

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for name, pred_mask in preds.items():
            try:
                # 始终保存为PNG格式，避免格式兼容性问题
                png_name = os.path.splitext(name)[0] + '.png'
                save_path = os.path.join(folder_path, png_name)
                
                # 确保是uint8类型
                if pred_mask.dtype != np.uint8:
                    pred_mask = pred_mask.astype(np.uint8)
                
                # 使用PIL保存图像
                img_pil = Image.fromarray(pred_mask)
                img_pil.save(save_path)
                print(f'成功保存预测结果到: {save_path}')
                
            except Exception as e:
                print(f'保存预测结果时出错: {str(e)}')
                traceback.print_exc()

    def infer(self, path, merged=True, save=True, folder_name=None):
        path = [path] if isinstance(path, str) else path

        preds = {}
        for p in path:
            try:
                # 获取文件名
                file_name = os.path.basename(p)
                print(f'处理图像: {file_name}')
                
                # 读取和预处理图像
                img, img_torch = self.read_and_preprocess(p)
                orig_h, orig_w = img.shape[:2]
                print(f"原始图像尺寸: {orig_w}x{orig_h}")
                
                # 模型推理
                with torch.no_grad():
                    pred_mask = self.swin_transunet.model(img_torch)
                    pred_mask = torch.sigmoid(pred_mask)
                    pred_mask = pred_mask.detach().cpu().numpy()
                
                # 提取预测掩码
                pred_np = pred_mask[0, 0]  # 取第一个批次的第一个通道
                print(f"预测掩码形状: {pred_np.shape}")
                
                # 二值化
                binary_mask = (pred_np > cfg.inference_threshold).astype(np.uint8) * 255
                
                # 创建掩码PIL图像
                mask_pil = Image.fromarray(binary_mask, mode='L')
                
                # 调整回原始大小
                mask_pil_resized = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                print(f"调整大小后的掩码尺寸: {mask_pil_resized.size}")
                
                # 准备输出
                if merged:
                    # 创建原始图像的PIL对象
                    img_pil = Image.fromarray(img)
                    
                    # 创建彩色掩码
                    color_mask = Image.new('RGB', (orig_w, orig_h), (0, 0, 0))
                    mask_data = mask_pil_resized.getdata()
                    color_data = []
                    
                    # 将掩码区域设置为红色
                    for pixel in mask_data:
                        if pixel > 0:
                            color_data.append((255, 0, 0))  # 红色
                        else:
                            color_data.append((0, 0, 0))  # 黑色
                    
                    color_mask.putdata(color_data)
                    
                    # 合成原始图像和掩码
                    result_pil = Image.blend(img_pil, color_mask, 0.5)
                    result_np = np.array(result_pil)
                else:
                    # 创建三通道掩码
                    mask_rgb = Image.new('RGB', (orig_w, orig_h), (0, 0, 0))
                    mask_data = mask_pil_resized.getdata()
                    rgb_data = []
                    
                    for pixel in mask_data:
                        rgb_data.append((pixel, pixel, pixel))  # 三个通道相同
                    
                    mask_rgb.putdata(rgb_data)
                    result_np = np.array(mask_rgb)
                
                # 保存结果
                preds[file_name] = result_np
                print(f"成功处理图像 {file_name}")
                
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
                traceback.print_exc()
                # 如果出错，创建一个空白图像
                if 'orig_w' in locals() and 'orig_h' in locals():
                    preds[file_name] = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                else:
                    # 使用默认尺寸
                    preds[file_name] = np.zeros((584, 565, 3), dtype=np.uint8)

        if save:
            self.save_preds(preds, folder_name)

        return preds


def main():
    parser = argparse.ArgumentParser(description='Swin TransUNet 推理')
    parser.add_argument('--model_path', type=str, default=cfg.swin_transunet.model_path, help='模型权重路径')
    parser.add_argument('--image_path', type=str, default="./DRIVE_PNG/test/images", help='输入图像路径，可以是单张图像或包含多张图像的文件夹')
    parser.add_argument('--output_folder', type=str, default="swin_transunet_results", help='输出文件夹名称，默认使用时间戳')
    parser.add_argument('--merged', action='store_true', default=False, help='是否将分割结果与原图合并')
    parser.add_argument('--no_save', action='store_true', default=False, help='是否不保存结果')
    args = parser.parse_args()

    # 检测GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 初始化推理类
    inference = SwinSegInference(args.model_path, device)

    # 处理输入路径
    if os.path.isdir(args.image_path):
        # 如果是文件夹，获取所有PNG图像文件
        image_files = []
        image_files.extend(glob.glob(os.path.join(args.image_path, '*.png')))
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
