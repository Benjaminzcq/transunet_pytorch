import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入项目模块
from utils.dataset import DentalDataset
from utils.utils import thresh_func, dice_coefficient
from model_swin_transunet import SwinTransUNetSeg
from model_transunet import TransUNetSeg
from config import cfg


class SimpleModelEvaluator:
    """
    简化版模型评估器：用于评估分割模型的性能并生成各种指标
    """
    def __init__(self, test_path, device):
        """
        初始化评估器
        
        Args:
            test_path: 测试数据集路径
            device: 计算设备（CPU或GPU）
        """
        self.device = device
        self.test_loader = self._load_dataset(test_path)
        
        # 创建结果目录
        if not os.path.exists('./evaluation_results'):
            os.makedirs('./evaluation_results')
    
    def _load_dataset(self, path):
        """加载测试数据集"""
        dataset = DentalDataset(path, transform=None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        return loader
    
    def _calculate_metrics(self, pred, target):
        """
        计算评估指标
        
        Args:
            pred: 预测掩码 (numpy array, 0-1值)
            target: 真实掩码 (numpy array, 0-1值)
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        # 确保输入是二值化的
        pred = (pred > 0.5).astype(np.float32)
        target = (target > 0.5).astype(np.float32)
        
        # 计算混淆矩阵元素
        tp = float(np.sum((pred == 1) & (target == 1)))
        tn = float(np.sum((pred == 0) & (target == 0)))
        fp = float(np.sum((pred == 1) & (target == 0)))
        fn = float(np.sum((pred == 0) & (target == 1)))
        
        # 计算评估指标
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # 计算IoU
        intersection = float(np.sum(pred * target))
        union = float(np.sum(pred) + np.sum(target) - intersection)
        iou = intersection / (union + 1e-10)
        
        # 计算Dice系数
        dice = 2 * intersection / (float(np.sum(pred)) + float(np.sum(target)) + 1e-10)
        
        # 构建混淆矩阵 - 使用纯测量值而非数组
        cm = [[tn, fp], [fn, tp]]
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice,
            'confusion_matrix': cm
        }
    
    def evaluate_model(self, model_wrapper, model_name):
        """
        评估模型性能
        
        Args:
            model_wrapper: 模型包装类 (TransUNetSeg 或 SwinTransUNetSeg)
            model_name: 模型名称
            
        Returns:
            dict: 包含评估结果的字典
        """
        # 访问实际的PyTorch模型
        model = model_wrapper.model
        model.eval()
        
        all_preds = []
        all_targets = []
        
        print(f"正在评估 {model_name}...")
        
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                img, mask = data['img'], data['mask']
                img = img.to(self.device)
                mask = mask.to(self.device)
                
                # 前向传播
                pred = model(img)
                
                # 将预测结果和真实标签转换为numpy数组
                pred_np = pred.cpu().numpy().squeeze()
                mask_np = mask.cpu().numpy().squeeze()
                
                # 二值化预测结果
                pred_binary = thresh_func(pred_np)
                
                all_preds.append(pred_binary)
                all_targets.append(mask_np)
        
        # 合并所有预测结果和真实标签
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        
        # 计算评估指标
        metrics = self._calculate_metrics(all_preds, all_targets)
        metrics['model_name'] = model_name
        
        return metrics
    
    def plot_confusion_matrix(self, results, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            results: 评估结果
            save_path: 保存路径
        """
        model_name = results['model_name']
        cm = results['confusion_matrix']
        
        # 使用纯文本方式生成混淆矩阵表示
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        # 创建一个简单的文本文件来代替图表
        cm_text = f"""混淆矩阵 - {model_name}

          | 预测负类 | 预测正类
--------------------------
真实负类 |   {int(tn)}    |    {int(fp)}
真实正类 |   {int(fn)}    |    {int(tp)}

准确率(Accuracy): {results['accuracy']:.4f}
精确率(Precision): {results['precision']:.4f}
召回率(Recall): {results['recall']:.4f}
F1分数: {results['f1']:.4f}
IoU: {results['iou']:.4f}
Dice系数: {results['dice']:.4f}
"""
        
        # 如果提供了保存路径，将文本保存到文件
        if save_path:
            # 将文件扩展名修改为.txt
            txt_path = os.path.splitext(save_path)[0] + '.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(cm_text)
            print(f"混淆矩阵文本已保存到: {txt_path}")
        
        # 打印到控制台
        print(cm_text)
    
    def save_results_to_csv(self, results_list, save_path):
        """
        将评估结果保存为CSV文件
        
        Args:
            results_list: 评估结果列表
            save_path: 保存路径
        """
        metrics = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']
        
        with open(save_path, 'w') as f:
            f.write(','.join(metrics) + '\n')
            
            for result in results_list:
                values = [str(result[metric]) if metric == 'model_name' else f"{result[metric]:.4f}" 
                          for metric in metrics]
                f.write(','.join(values) + '\n')
        
        print(f"评估结果已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="医学图像分割模型评估")
    parser.add_argument('--test_path', type=str, default="./DRIVE_PNG/test", help='测试数据集路径')
    parser.add_argument('--transunet_model', type=str, default="./saved_models/transunet_model.pth", help='TransUNet模型路径')
    parser.add_argument('--swin_model', type=str, default="./saved_models/swin_transunet_model.pth", help='Swin TransUNet模型路径')
    parser.add_argument('--output_dir', type=str, default="./evaluation_results", help='评估结果输出目录')
    args = parser.parse_args()
    
    # 检测设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 初始化评估器
    evaluator = SimpleModelEvaluator(args.test_path, device)
    
    results_list = []
    
    # 评估TransUNet模型
    if os.path.exists(args.transunet_model):
        transunet = TransUNetSeg(device)
        transunet.load_model(args.transunet_model)
        transunet_results = evaluator.evaluate_model(transunet, 'TransUNet')
        results_list.append(transunet_results)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.output_dir, 'transunet_confusion_matrix.txt')
        evaluator.plot_confusion_matrix(transunet_results, cm_path)
    else:
        print(f"警告: TransUNet模型文件 {args.transunet_model} 不存在")
    
    # 评估Swin TransUNet模型
    if os.path.exists(args.swin_model):
        swin_transunet = SwinTransUNetSeg(device)
        swin_transunet.load_model(args.swin_model)
        swin_results = evaluator.evaluate_model(swin_transunet, 'Swin TransUNet')
        results_list.append(swin_results)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.output_dir, 'swin_transunet_confusion_matrix.txt')
        evaluator.plot_confusion_matrix(swin_results, cm_path)
    else:
        print(f"警告: Swin TransUNet模型文件 {args.swin_model} 不存在")
    
    # 保存评估结果
    if results_list:
        csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        evaluator.save_results_to_csv(results_list, csv_path)
    else:
        print("错误: 没有可用的模型进行评估")


if __name__ == '__main__':
    main()
