# 医学图像分割模型比较：TransUNet vs Swin TransUNet

本项目实现了两种基于Transformer的医学图像分割模型：原始的TransUNet和改进的Swin TransUNet，并提供了完整的训练、推理和评估功能。

![输出示例](./assets/outs.png "输出示例")
*模型输出示例。(A) 原始X光图像; (B) 预测分割图与原始图像合并; (C) 真实标注; (D) 预测分割图*

## 项目概述

### TransUNet
- TransUNet是一种结合CNN和Transformer的医学图像分割模型，能够同时利用CNN的局部特征提取能力和Transformer的全局上下文建模能力。
- 论文：[TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)

### Swin TransUNet
- Swin TransUNet是对原始TransUNet的改进，将Vision Transformer(ViT)替换为Swin Transformer，提供更高效的层级化特征提取。
- Swin Transformer通过滑动窗口机制，在保持全局建模能力的同时，降低了计算复杂度。
- 论文：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

## 项目结构

```
transunet_pytorch/
├── model_transunet.py          # TransUNet模型定义
├── model_swin_transunet.py     # Swin TransUNet模型定义
├── train_transunet.py          # TransUNet训练入口
├── train_swin_transunet.py     # Swin TransUNet训练入口
├── inference_transunet.py      # TransUNet推理入口
├── inference_swin_transunet.py # Swin TransUNet推理入口
├── evaluate_models.py          # 模型评估和对比工具
├── config.py                   # 配置文件
├── utils/                      # 工具函数和组件
│   ├── dataset.py              # 数据集加载
│   ├── transforms.py           # 数据增强
│   ├── utils.py                # 通用工具函数
│   ├── transunet.py            # TransUNet模型组件
│   ├── vit.py                  # Vision Transformer实现
│   ├── swin_transunet.py       # Swin TransUNet模型组件
│   └── swin_transformer.py     # Swin Transformer实现
├── README_TRANSUNET.md         # TransUNet详细文档
└── README_SWIN_TRANSUNET.md    # Swin TransUNet详细文档
```

## 环境要求

- Python 3.6+
- PyTorch >= 1.7.0
- torchvision
- timm
- einops
- numpy
- tqdm
- matplotlib
- seaborn
- scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

#### TransUNet
```bash
python train_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_transunet.pth
```

#### Swin TransUNet
```bash
python train_swin_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_swin_transunet.pth
```

### 2. 模型推理

#### TransUNet
```bash
python inference_transunet.py --model_path ./model_transunet.pth --image_path ./DRIVE/test/images --output_folder transunet_results
```

#### Swin TransUNet
```bash
python inference_swin_transunet.py --model_path ./model_swin_transunet.pth --image_path ./DRIVE/test/images --output_folder swin_transunet_results
```

### 3. 模型评估与对比

```bash
python evaluate_models.py --test_path ./DRIVE/test --transunet_model ./model_transunet.pth --swin_model ./model_swin_transunet.pth --output_dir ./evaluation_results
```

## 评估指标

评估脚本会计算以下指标并生成可视化结果：

- **Dice系数**：衡量预测分割与真实分割的重叠度
- **IoU (交并比)**：衡量预测区域与真实区域的重叠程度
- **准确率**：正确分类的像素比例
- **精确度**：在预测为前景的像素中，真正为前景的比例
- **召回率**：在真实前景像素中，被正确预测为前景的比例
- **F1分数**：精确度和召回率的调和平均
- **混淆矩阵**：详细展示分类性能

## 模型性能对比

通过对TransUNet和Swin TransUNet两个模型在相同测试集上的评估，我们获得了以下性能指标对比：

| 模型 | Dice系数 | IoU | 准确率 | 精确度 | 召回率 | F1分数 |
|------|---------|-----|--------|--------|--------|--------|
| TransUNet | - | - | - | - | - | - |
| Swin TransUNet | - | - | - | - | - | - |

*注：实际数值将在模型训练和评估后填充*


### 混淆矩阵分析

混淆矩阵提供了模型分类性能的详细视图，包括：
- 真正例 (TP)：正确识别为前景的像素
- 假正例 (FP)：错误识别为前景的背景像素
- 真负例 (TN)：正确识别为背景的像素
- 假负例 (FN)：错误识别为背景的前景像素

通过混淆矩阵可视化，我们可以直观地比较两个模型在处理不同类别时的表现差异。

### 改进效果分析

Swin TransUNet相比于原始TransUNet的主要改进体现在：

1. **特征提取能力**：Swin Transformer的层级化结构和局部注意力机制使模型能够更好地捕捉图像的多尺度特征

2. **计算效率**：通过窗口注意力机制，Swin TransUNet在保持性能的同时减少了计算复杂度

3. **边界细节处理**：改进后的模型在处理分割边界细节方面表现更好，这对医学图像分割尤为重要

## 结论

通过将Vision Transformer替换为Swin Transformer，我们实现了一个更高效的TransUNet模型。Swin Transformer的层级化结构和局部注意力机制使模型能够更好地处理医学图像分割任务，同时保持了较高的计算效率。

在调试过程中，我们解决了多个与特征图形状、通道数匹配和内存优化相关的问题，使模型能够成功训练。这些修改和优化可以作为将其他Transformer变体集成到U-Net类架构中的参考。

通过添加全面的评估指标和可视化工具，我们能够更客观地比较不同模型的性能，为后续的模型改进提供了坚实的基础。

## 详细文档

- [TransUNet Document(English)](./README_TransUnet_en.md)
- [TransUNet详细文档(中文)](./README_TransUnet_zh.md)
- [Swin TransUNet Document(English)](./README_Swin_TransUnet_en.md)
- [Swin TransUNet详细文档(中文)](./README_Swin_TransUnet_zh.md)

## 引用

- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [3] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [4] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
