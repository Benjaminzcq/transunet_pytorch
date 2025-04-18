# Swin TransUNet: 基于Swin Transformer的改进型医学图像分割模型

本文档详细介绍了Swin TransUNet模型的实现、训练和使用方法。Swin TransUNet是对原始TransUNet的改进版本，将Vision Transformer(ViT)替换为Swin Transformer，提供更高效的层级化特征提取。

## 模型概述

Swin TransUNet是将原始TransUNet模型中的Vision Transformer(ViT)替换为Swin Transformer的实现。Swin Transformer是一种基于滑动窗口的层级化Transformer，相比于原始ViT，它能够更好地处理图像的局部信息和层级特征，同时具有更高的计算效率。

## 模型架构

Swin TransUNet保持了TransUNet的整体U型结构，但在编码器部分使用了Swin Transformer替代原始的Vision Transformer。主要改进包括：

1. **滑动窗口注意力机制**：通过在局部窗口内计算自注意力，降低了计算复杂度
2. **层级化特征提取**：通过逐步合并图像块，形成层级化的表示
3. **相对位置编码**：使用相对位置编码代替绝对位置编码，提高了泛化能力

## 文件结构与说明

### 核心文件

1. **utils/swin_transformer.py**
   - 实现了Swin Transformer的核心组件，包括窗口划分、窗口注意力、Patch合并等
   - 包含了处理特征图尺寸不被窗口大小整除的逻辑
   - 实现了自适应形状调整，确保特征图能够正确传递给后续层

2. **utils/swin_transunet.py**
   - 实现了基于Swin Transformer的TransUNet模型
   - 包含编码器(Encoder)和解码器(Decoder)结构
   - 实现了通道适配器，用于处理Swin Transformer输出与后续卷积层的通道数匹配
   - 包含特征图尺寸自适应调整逻辑，确保Skip Connection正确工作

3. **model_swin_transunet.py**
   - 定义了Swin TransUNet模型的封装类，包含初始化、加载、训练和测试方法

4. **train_swin_transunet.py**
   - 训练脚本的入口点
   - 设置训练参数、数据加载和训练循环

5. **inference_swin_transunet.py**
   - 推理脚本的入口点
   - 实现了图像预处理、模型推理和结果保存功能

6. **config.py**
   - 配置文件，定义了模型参数、训练参数等

## 主要改进内容

### 1. Swin Transformer实现

- **窗口分割与合并**：实现了能够处理任意尺寸特征图的窗口分割和合并函数
  ```python
  def window_partition(x, window_size):
      # 支持特征图尺寸不被窗口大小整除的情况
      B, H, W, C = x.shape
      pad_h = (window_size - H % window_size) % window_size
      pad_w = (window_size - W % window_size) % window_size
      if pad_h > 0 or pad_w > 0:
          x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
      # 窗口分割逻辑...
  ```

- **形状自适应**：在SwinTransformerBlock中添加了输入形状自适应调整
  ```python
  # 如果输入特征形状与预期不符，将其调整为匹配的尺寸
  if L != H * W:
      # 调整形状逻辑...
  ```

### 2. 通道适配器

- 在Encoder类中添加了通道适配器，用于处理Swin Transformer输出的特征图与后续卷积层的通道数不匹配问题
  ```python
  # 通道适配器，用于处理Swin Transformer输出的特征图与后续卷积层的通道数不匹配问题
  self.input_channels = 4096  # Swin Transformer输出通道数
  self.num_features = out_channels * 8  # 期望的通道数
  self.channel_adapter = nn.Conv2d(self.input_channels, self.num_features, kernel_size=1, bias=False)
  ```

### 3. 解码器改进

- 在DecoderBottleneck类中添加了空间尺寸自适应调整功能，确保特征图拼接时尺寸匹配
  ```python
  # 检查空间尺寸是否匹配
  if x.shape[2:] != x_concat.shape[2:]:
      # 如果空间尺寸不匹配，将x调整为x_concat的尺寸
      x = nn.functional.interpolate(x, size=x_concat.shape[2:], mode='bilinear', align_corners=True)
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

使用以下命令训练Swin TransUNet模型：

```bash
python train_swin_transunet.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
```

参数说明：
- `--train_path`：训练数据集路径，默认为"./DRIVE/training"
- `--test_path`：测试数据集路径，默认为"./DRIVE/test"
- `--model_path`：模型保存路径，默认为"./model_swin_transunet.pth"

示例：
```bash
python train_swin_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_swin_transunet.pth
```

## 推理预测

使用训练好的模型进行图像分割预测：

```bash
python inference_swin_transunet.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
```

参数说明：
- `--model_path`：模型权重路径，默认为"./model_swin_transunet.pth"
- `--image_path`：输入图像路径，可以是单张图像或包含多张图像的文件夹，默认为"./DRIVE/test/images"
- `--output_folder`：输出文件夹名称，默认为"swin_transunet_results"
- `--merged`：是否将分割结果与原图合并，默认为True
- `--no_save`：设置此参数将不保存结果

示例：
```bash
python inference_swin_transunet.py --model_path ./model_swin_transunet.pth --image_path ./DRIVE/test/images --output_folder swin_transunet_results
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
python evaluate_models.py --test_path ./DRIVE/test --swin_model ./model_swin_transunet.pth --output_dir ./evaluation_results
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
3. 性能对比图（如果同时评估了TransUNet）
4. 包含所有指标的CSV文件

## 模型参数调整

如果遇到内存不足问题，可以尝试以下调整：

1. 减小批次大小：在config.py中修改batch_size参数
   ```python
   cfg.batch_size = 2  # 减小批次大小以减少显存使用
   ```

2. 降低模型复杂度：在config.py中修改Swin Transformer的配置
   ```python
   cfg.swin_transunet.depths = [2, 2, 2, 2]  # 减少Swin Transformer的深度
   cfg.swin_transunet.num_heads = [2, 4, 8, 8]  # 减少注意力头数
   ```

## 引用

- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [3] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [4] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
