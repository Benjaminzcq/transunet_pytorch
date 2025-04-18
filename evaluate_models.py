import os
import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

# 导入项目模块
from utils.dataset import DentalDataset
from utils.utils import thresh_func, dice_coefficient
from model_swin_transunet import SwinTransUNetSeg
from model_transunet import TransUNetSeg
from config import cfg


class ModelEvaluator:
    """
    模型评估器：用于评估分割模型的性能并生成各种指标
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
    
    def _calculate_iou(self, pred, target):
        """
        计算IoU (Intersection over Union)
        
        Args:
            pred: 预测掩码
            target: 真实掩码
            
        Returns:
            float: IoU值
        """
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        if union == 0:
            return 0
        return intersection / union
    
    def evaluate_model(self, model, model_name):
        """
        评估模型性能
        
        Args:
            model: 待评估的模型
            model_name: 模型名称
            
        Returns:
            dict: 包含各项评估指标的字典
        """
        model.model.eval()
        
        # 初始化指标
        dice_scores = []
        iou_scores = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # 用于计算混淆矩阵
        all_preds = []
        all_targets = []
        
        print(f"\n正在评估 {model_name} 模型...")
        
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                img, mask = data['img'], data['mask']
                img = img.to(self.device)
                mask = mask.to(self.device)
                
                # 模型预测
                pred_mask = model.model(img)
                pred_mask = torch.sigmoid(pred_mask)
                
                # 转换为二值掩码
                pred_np = pred_mask.detach().cpu().numpy().squeeze()
                pred_binary = thresh_func(pred_np.copy(), thresh=cfg.inference_threshold)
                
                # 转换真实掩码为numpy数组
                target_np = mask.detach().cpu().numpy().squeeze()
                
                # 计算Dice系数
                dice = dice_coefficient(pred_binary, target_np)
                dice_scores.append(dice)
                
                # 计算IoU
                iou = self._calculate_iou(pred_binary, target_np)
                iou_scores.append(iou)
                
                # 将掩码展平用于计算其他指标
                pred_flat = pred_binary.flatten()
                target_flat = target_np.flatten()
                
                # 计算准确率
                acc = accuracy_score(target_flat, pred_flat)
                accuracies.append(acc)
                
                # 计算精确度、召回率和F1分数
                precision = precision_score(target_flat, pred_flat, zero_division=0)
                recall = recall_score(target_flat, pred_flat, zero_division=0)
                f1 = f1_score(target_flat, pred_flat, zero_division=0)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
                # 收集用于混淆矩阵的数据
                all_preds.extend(pred_flat)
                all_targets.extend(target_flat)
        
        # 计算平均指标
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        # 保存结果
        results = {
            'model_name': model_name,
            'dice': avg_dice,
            'iou': avg_iou,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'confusion_matrix': cm
        }
        
        # 打印结果
        print(f"\n{model_name} 评估结果:")
        print(f"Dice系数: {avg_dice:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"准确率: {avg_accuracy:.4f}")
        print(f"精确度: {avg_precision:.4f}")
        print(f"召回率: {avg_recall:.4f}")
        print(f"F1分数: {avg_f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, results, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            results: 评估结果字典
            save_path: 保存路径
        """
        model_name = results['model_name']
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 设置坐标轴标签
        plt.xticks([0, 1], ['背景', '前景'])
        plt.yticks([0, 1], ['背景', '前景'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{model_name} 混淆矩阵')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, results_list, save_path=None):
        """
        绘制不同模型的指标对比图
        
        Args:
            results_list: 包含多个模型评估结果的列表
            save_path: 保存路径
        """
        metrics = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']
        model_names = [r['model_name'] for r in results_list]
        
        # 准备数据
        metrics_data = {metric: [r[metric] for r in results_list] for metric in metrics}
        
        # 绘制柱状图
        plt.figure(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            values = [metrics_data[metric][i] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model_name)
        
        plt.xlabel('评估指标')
        plt.ylabel('分数')
        plt.title('模型性能对比')
        plt.xticks(x + width/2, [m.capitalize() for m in metrics])
        plt.legend()
        plt.ylim(0, 1)
        
        # 添加数值标签
        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                value = metrics_data[metric][i]
                plt.text(j + i*width, value + 0.01, f'{value:.3f}', 
                         ha='center', va='bottom', fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")
        
        plt.close()
    
    def save_results_to_csv(self, results_list, save_path):
        """
        将评估结果保存为CSV文件
        
        Args:
            results_list: 包含多个模型评估结果的列表
            save_path: 保存路径
        """
        metrics = ['model_name', 'dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']
        
        with open(save_path, 'w') as f:
            # 写入表头
            f.write(','.join(metrics) + '\n')
            
            # 写入每个模型的结果
            for result in results_list:
                values = [str(result[metric]) if metric == 'model_name' else f"{result[metric]:.4f}" 
                          for metric in metrics]
                f.write(','.join(values) + '\n')
        
        print(f"评估结果已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="医学图像分割模型评估")
    parser.add_argument('--test_path', type=str, default="./DRIVE_PNG/test", help='测试数据集路径')
    parser.add_argument('--transunet_model', type=str, default="./transunet_model.pth", help='TransUNet模型路径')
    parser.add_argument('--swin_model', type=str, default="./swin_transunet_model.pth", help='Swin TransUNet模型路径')
    parser.add_argument('--output_dir', type=str, default="./evaluation_results", help='评估结果输出目录')
    args = parser.parse_args()
    
    # 检测设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 初始化评估器
    evaluator = ModelEvaluator(args.test_path, device)
    
    results_list = []
    
    # 评估TransUNet模型
    if os.path.exists(args.transunet_model):
        transunet = TransUNetSeg(device)
        transunet.load_model(args.transunet_model)
        transunet_results = evaluator.evaluate_model(transunet, "TransUNet")
        results_list.append(transunet_results)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.output_dir, "transunet_confusion_matrix.png")
        evaluator.plot_confusion_matrix(transunet_results, cm_path)
    else:
        print(f"警告: TransUNet模型文件 {args.transunet_model} 不存在，跳过评估")
    
    # 评估Swin TransUNet模型
    if os.path.exists(args.swin_model):
        swin_transunet = SwinTransUNetSeg(device)
        swin_transunet.load_model(args.swin_model)
        swin_results = evaluator.evaluate_model(swin_transunet, "Swin TransUNet")
        results_list.append(swin_results)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.output_dir, "swin_transunet_confusion_matrix.png")
        evaluator.plot_confusion_matrix(swin_results, cm_path)
    else:
        print(f"警告: Swin TransUNet模型文件 {args.swin_model} 不存在，跳过评估")
    
    # 如果两个模型都已评估，绘制对比图
    if len(results_list) > 1:
        comparison_path = os.path.join(args.output_dir, "models_comparison.png")
        evaluator.plot_metrics_comparison(results_list, comparison_path)
        
        # 保存结果到CSV
        csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
        evaluator.save_results_to_csv(results_list, csv_path)
    

if __name__ == '__main__':
    main()
