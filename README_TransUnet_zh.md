# TransUNet: 基于Transformer的医学图像分割模型

本文档详细介绍了TransUNet模型的实现、训练和使用方法。TransUNet是一种结合CNN和Transformer的医学图像分割模型，能够同时利用CNN的局部特征提取能力和Transformer的全局上下文建模能力。

![输出示例](./assets/outs.png "输出示例")
*TransUNet模型输出示例。(A) 原始X光图像; (B) 预测分割图与原始图像合并; (C) 真实标注; (D) 预测分割图*

## 模型概述

- TransUNet在各种医学图像分割任务中表现出色，特别是在处理需要全局上下文理解的复杂医学图像时。
- 模型采用U型架构，结合了CNN的高分辨率空间信息和Transformer的全局上下文编码能力。
- 相比传统的U-Net，TransUNet能够更好地处理长距离依赖关系，提高分割精度。

## 模型架构

![模型架构](./assets/arch.png "模型架构")

*TransUNet架构图（来自原始论文）*

TransUNet的主要组件包括：
1. **CNN编码器**：使用ResNet作为特征提取的骨干网络
2. **Transformer编码器**：将CNN提取的特征转换为序列，并通过多头自注意力机制捕获全局依赖关系
3. **解码器**：通过上采样和跳跃连接恢复空间细节，生成最终的分割图

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

## 数据集准备

本项目支持多种图像格式的数据集，包括但不限于：
- 图像文件：支持.jpg、.png、.bmp、.gif、.tif、.jpeg等格式
- 掩码文件：支持.jpg、.png、.bmp、.gif、.tif、.jpeg等格式

数据集目录结构应如下：
```
数据集根目录/
  ├── images/       # 存放原始图像
  └── mask/         # 存放对应的掩码图像
```

注意：
- 对于DRIVE数据集，图像文件格式为XX_training.tif，掩码文件格式为XX_training_mask.gif
- 程序会自动匹配图像和掩码文件，确保它们能够正确对应
- 对于GIF格式的掩码文件，程序使用PIL库进行读取，以解决OpenCV无法正确读取GIF文件的问题

## 模型训练

使用以下命令训练TransUNet模型：

```bash
python train_transunet.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
```

参数说明：
- `--train_path`：训练数据集路径，默认为"./DRIVE/training"
- `--test_path`：测试数据集路径，默认为"./DRIVE/test"
- `--model_path`：模型保存路径，默认为"./model_transunet.pth"

示例：
```bash
python train_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_transunet.pth
```

## 模型推理

使用训练好的模型进行图像分割预测：

```bash
python inference_transunet.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
```

参数说明：
- `--model_path`：模型权重路径，默认为"./model_transunet.pth"
- `--image_path`：输入图像路径，可以是单张图像或包含多张图像的文件夹，默认为"./DRIVE/test/images"
- `--output_folder`：输出文件夹名称，默认为"transunet_results"
- `--merged`：是否将分割结果与原图合并，默认为True
- `--no_save`：设置此参数将不保存结果

示例：
```bash
python inference_transunet.py --model_path ./model_transunet.pth --image_path ./DRIVE/test/images --output_folder transunet_results
```

## 模型评估

评估模型性能并生成详细的评估指标报告：

```bash
python evaluate_models.py [--test_path TEST_PATH] [--transunet_model TRANSUNET_MODEL_PATH] [--swin_model SWIN_MODEL_PATH] [--output_dir OUTPUT_DIR]
```

参数说明：
- `--test_path`：测试数据集路径，默认为"./DRIVE/test"
- `--transunet_model`：TransUNet模型路径，默认为"./model_transunet.pth"
- `--swin_model`：Swin TransUNet模型路径，默认为"./model_swin_transunet.pth"
- `--output_dir`：评估结果输出目录，默认为"./evaluation_results"

示例：
```bash
python evaluate_models.py --test_path ./DRIVE/test --transunet_model ./model_transunet.pth --output_dir ./evaluation_results
```

## 评估指标

评估脚本会计算以下指标：
- **Dice系数**：衡量预测分割与真实分割的重叠度
- **IoU (交并比)**：衡量预测区域与真实区域的重叠程度
- **准确率**：正确分类的像素比例
- **精确度**：在预测为前景的像素中，真正为前景的比例
- **召回率**：在真实前景像素中，被正确预测为前景的比例
- **F1分数**：精确度和召回率的调和平均

评估结果将以以下形式输出：
1. 控制台打印的详细指标
2. 混淆矩阵可视化
3. 性能对比图（如果同时评估了Swin TransUNet）
4. 包含所有指标的CSV文件

## 模型参数调整

如果遇到内存不足问题，可以尝试以下调整：

1. 减小批次大小：在config.py中修改batch_size参数
   ```python
   cfg.batch_size = 2  # 减小批次大小以减少显存使用
   ```

2. 降低模型复杂度：在config.py中修改TransUNet的配置
   ```python
   cfg.transunet.out_channels = 64  # 减小输出通道数
   cfg.transunet.mlp_dim = 256      # 减小MLP维度
   ```

## 引用

- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [3] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
