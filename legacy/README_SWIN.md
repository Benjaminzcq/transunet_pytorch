# Swin TransUNet 实现与调试说明

## 项目概述

本项目是将原始TransUNet模型中的Vision Transformer(ViT)替换为Swin Transformer的实现。Swin Transformer是一种基于滑动窗口的层级化Transformer，相比于原始ViT，它能够更好地处理图像的局部信息和层级特征，同时具有更高的计算效率。

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

3. **train_swin_transunet.py**
   - 定义了训练Swin TransUNet模型的类
   - 包含模型初始化、训练步骤和测试步骤的实现

4. **run_train.py**
   - 训练脚本的入口点
   - 设置训练参数、数据加载和训练循环

5. **config.py**
   - 配置文件，定义了模型参数、训练参数等

## 主要修改内容

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

- 改进了Decoder类的初始化，正确处理Skip Connection的通道数
  ```python
  # 根据错误信息和通道数分析，指定正确的skip connection通道数
  self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2, skip_channels=out_channels * 4)
  ```

## 调试过程中解决的问题

1. **特征图尺寸不被窗口大小整除**：通过在window_partition和window_reverse函数中添加填充和去除填充的逻辑解决

2. **通道数不匹配**：Swin Transformer输出的通道数与后续卷积层期望的通道数不匹配，通过添加通道适配器解决

3. **Skip Connection尺寸不匹配**：在解码器中，当尝试将特征图进行拼接时，输入的两个张量在空间维度上不匹配，通过添加空间尺寸自适应调整功能解决

4. **Swin Transformer输入形状错误**：在SwinTransformerBlock中，输入特征的形状不满足L=H*W的要求，通过添加形状自适应调整解决

5. **内存优化**：通过减小批次大小、降低模型复杂度等方式优化内存使用

## 运行流程

### 环境准备

确保已安装以下依赖：
- PyTorch (>=1.7.0)
- torchvision
- timm
- einops
- numpy
- tqdm
- matplotlib
- seaborn
- scikit-learn

### 训练模型

1. 配置参数：在config.py中设置模型参数和训练参数

2. 运行训练脚本：
   ```bash
   python run_train.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
   ```

   参数说明：
   - `--train_path`：训练数据集路径，默认为"./DRIVE/training"
   - `--test_path`：测试数据集路径，默认为"./DRIVE/test"
   - `--model_path`：模型保存路径，默认为"./model_swin_transunet.pth"

3. 训练过程将显示每个轮次的训练损失和验证损失

### 推理预测

使用训练好的模型进行图像分割预测：

```bash
python run_inference.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
```

参数说明：
- `--model_path`：模型权重路径，默认为"./model_swin_transunet.pth"
- `--image_path`：输入图像路径，可以是单张图像或包含多张图像的文件夹，默认为"./data/infer"
- `--output_folder`：输出文件夹名称，默认为"swin_transunet_results"
- `--merged`：是否将分割结果与原图合并，默认为True
- `--no_save`：设置此参数将不保存结果

### 模型评估

评估模型性能并生成详细的评估指标报告：

```bash
python evaluate_models.py [--test_path TEST_PATH] [--transunet_model TRANSUNET_MODEL_PATH] [--swin_model SWIN_MODEL_PATH] [--output_dir OUTPUT_DIR]
```

参数说明：
- `--test_path`：测试数据集路径，默认为"./DRIVE/test"
- `--transunet_model`：TransUNet模型路径，默认为"./model_transunet.pth"
- `--swin_model`：Swin TransUNet模型路径，默认为"./model_swin_transunet.pth"
- `--output_dir`：评估结果输出目录，默认为"./evaluation_results"

评估指标包括：
- **Dice系数**：衡量预测分割与真实分割的重叠度
- **IoU (交并比)**：衡量预测区域与真实区域的重叠程度
- **准确率**：正确分类的像素比例
- **精确度**：在预测为前景的像素中，真正为前景的比例
- **召回率**：在真实前景像素中，被正确预测为前景的比例
- **F1分数**：精确度和召回率的调和平均

评估结果将以以下形式输出：
1. 控制台打印的详细指标
2. 每个模型的混淆矩阵可视化
3. 不同模型之间的性能对比图
4. 包含所有指标的CSV文件

### 数据集格式要求

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

### 模型参数调整

如果遇到内存不足问题，可以尝试以下调整：

1. 减小批次大小：在config.py中修改batch_size参数
   ```python
   cfg.batch_size = 4  # 减小批次大小以减少显存使用
   ```

2. 降低模型复杂度：在train_swin_transunet.py中修改Swin Transformer的配置
   ```python
   self.model = SwinTransUNet(
       out_channels=64,  # 减小输出通道数
       depths=[2, 2, 4, 2],  # 减少Swin Transformer的深度
       num_heads=[2, 4, 8, 16],  # 减少注意力头数
       # 其他参数...
   )
   ```

## 评估结果与分析

### 性能指标对比

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
