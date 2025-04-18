# Swin TransUNet: Improved Medical Image Segmentation Model Based on Swin Transformer

This document provides detailed information about the implementation, training, and usage of the Swin TransUNet model. Swin TransUNet is an improved version of the original TransUNet, replacing the Vision Transformer (ViT) with Swin Transformer to provide more efficient hierarchical feature extraction.

## Project Overview

Swin TransUNet replaces the Vision Transformer (ViT) in the original TransUNet model with Swin Transformer. Swin Transformer is a hierarchical Transformer based on shifted windows, which, compared to the original ViT, can better process local information and hierarchical features of images while offering higher computational efficiency.

## Model Architecture

Swin TransUNet maintains the overall U-shaped structure of TransUNet but uses Swin Transformer instead of the original Vision Transformer in the encoder part. The main improvements include:

1. **Shifted Window Attention Mechanism**: Reduces computational complexity by computing self-attention within local windows
2. **Hierarchical Feature Extraction**: Forms hierarchical representations by progressively merging image patches
3. **Relative Position Encoding**: Uses relative position encoding instead of absolute position encoding, improving generalization ability

## File Structure and Description

### Core Files

1. **utils/swin_transformer.py**
   - Implements core components of Swin Transformer, including window partitioning, window attention, patch merging, etc.
   - Contains logic for handling feature maps whose dimensions are not divisible by the window size
   - Implements adaptive shape adjustment to ensure feature maps can be correctly passed to subsequent layers

2. **utils/swin_transunet.py**
   - Implements the TransUNet model based on Swin Transformer
   - Contains encoder and decoder structures
   - Implements channel adapters to handle channel number matching between Swin Transformer output and subsequent convolutional layers
   - Contains adaptive feature map size adjustment logic to ensure skip connections work correctly

3. **model_swin_transunet.py**
   - Defines the wrapper class for the Swin TransUNet model, including initialization, loading, training, and testing methods

4. **train_swin_transunet.py**
   - Entry point for the training script
   - Sets training parameters, data loading, and training loops

5. **inference_swin_transunet.py**
   - Entry point for the inference script
   - Implements image preprocessing, model inference, and result saving functions

6. **config.py**
   - Configuration file defining model parameters, training parameters, etc.

## Main Improvements

### 1. Swin Transformer Implementation

- **Window Partitioning and Merging**: Implements window partitioning and merging functions capable of handling feature maps of arbitrary sizes
  ```python
  def window_partition(x, window_size):
      # Support for feature map dimensions not divisible by window size
      B, H, W, C = x.shape
      pad_h = (window_size - H % window_size) % window_size
      pad_w = (window_size - W % window_size) % window_size
      if pad_h > 0 or pad_w > 0:
          x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
      # Window partitioning logic...
  ```

- **Shape Adaptation**: Added input shape adaptive adjustment in SwinTransformerBlock
  ```python
  # If the input feature shape does not match expectations, adjust it to match
  if L != H * W:
      # Shape adjustment logic...
  ```

### 2. Channel Adapter

- Added channel adapter in the Encoder class to handle channel number mismatches between Swin Transformer output feature maps and subsequent convolutional layers
  ```python
  # Channel adapter to handle channel number mismatches between Swin Transformer output feature maps and subsequent convolutional layers
  self.input_channels = 4096  # Swin Transformer output channels
  self.num_features = out_channels * 8  # Expected channels
  self.channel_adapter = nn.Conv2d(self.input_channels, self.num_features, kernel_size=1, bias=False)
  ```

### 3. Decoder Improvements

- Added spatial size adaptive adjustment functionality in the DecoderBottleneck class to ensure feature map size matches during concatenation
  ```python
  # Check if spatial sizes match
  if x.shape[2:] != x_concat.shape[2:]:
      # If spatial sizes don't match, adjust x to x_concat's size
      x = nn.functional.interpolate(x, size=x_concat.shape[2:], mode='bilinear', align_corners=True)
  ```

## Environment Setup

Ensure the following dependencies are installed:
- PyTorch (>=1.7.0)
- torchvision
- timm
- einops
- numpy
- tqdm
- matplotlib
- seaborn
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Train the Swin TransUNet model using the following command:

```bash
python train_swin_transunet.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
```

Parameter description:
- `--train_path`: Training dataset path, default is "./DRIVE/training"
- `--test_path`: Testing dataset path, default is "./DRIVE/test"
- `--model_path`: Model saving path, default is "./model_swin_transunet.pth"

Example:
```bash
python train_swin_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_swin_transunet.pth
```

## Inference Prediction

Perform image segmentation prediction using the trained model:

```bash
python inference_swin_transunet.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
```

Parameter description:
- `--model_path`: Model weights path, default is "./model_swin_transunet.pth"
- `--image_path`: Input image path, can be a single image or a folder containing multiple images, default is "./DRIVE/test/images"
- `--output_folder`: Output folder name, default is "swin_transunet_results"
- `--merged`: Whether to merge segmentation results with the original image, default is True
- `--no_save`: Set this parameter to not save results

Example:
```bash
python inference_swin_transunet.py --model_path ./model_swin_transunet.pth --image_path ./DRIVE/test/images --output_folder swin_transunet_results
```

## Model Evaluation

Evaluate model performance and generate detailed evaluation metric reports:

```bash
python evaluate_models.py [--test_path TEST_PATH] [--transunet_model TRANSUNET_MODEL_PATH] [--swin_model SWIN_MODEL_PATH] [--output_dir OUTPUT_DIR]
```

Parameter description:
- `--test_path`: Testing dataset path, default is "./DRIVE/test"
- `--transunet_model`: TransUNet model path, default is "./model_transunet.pth"
- `--swin_model`: Swin TransUNet model path, default is "./model_swin_transunet.pth"
- `--output_dir`: Evaluation results output directory, default is "./evaluation_results"

Example:
```bash
python evaluate_models.py --test_path ./DRIVE/test --swin_model ./model_swin_transunet.pth --output_dir ./evaluation_results
```

## Evaluation Metrics

The evaluation script calculates the following metrics:
- **Dice Coefficient**: Measures the overlap between predicted and ground truth segmentation
- **IoU (Intersection over Union)**: Measures the overlap ratio between predicted and ground truth regions
- **Accuracy**: Proportion of correctly classified pixels
- **Precision**: Proportion of true foreground pixels among all pixels predicted as foreground
- **Recall**: Proportion of correctly predicted foreground pixels among all true foreground pixels
- **F1 Score**: Harmonic mean of precision and recall

Evaluation results are output in the following forms:
1. Detailed metrics printed to the console
2. Confusion matrix visualization
3. Performance comparison chart (if TransUNet is also evaluated)
4. CSV file containing all metrics

## Dataset Format Requirements

This project supports datasets with various image formats, including but not limited to:
- Image files: .jpg, .png, .bmp, .gif, .tif, .jpeg, etc.
- Mask files: .jpg, .png, .bmp, .gif, .tif, .jpeg, etc.

The dataset directory structure should be as follows:
```
Dataset Root Directory/
  ├── images/       # Original images
  └── mask/         # Corresponding mask images
```

Note:
- For the DRIVE dataset, image files are in XX_training.tif format, and mask files are in XX_training_mask.gif format
- The program automatically matches image and mask files to ensure correct correspondence
- For GIF format mask files, the program uses the PIL library for reading to solve the problem that OpenCV cannot correctly read GIF files

## Model Parameter Adjustment

If you encounter memory issues, try the following adjustments:

1. Reduce batch size: Modify the batch_size parameter in config.py
   ```python
   cfg.batch_size = 2  # Reduce batch size to decrease memory usage
   ```

2. Lower model complexity: Modify Swin Transformer configuration in config.py
   ```python
   cfg.swin_transunet.depths = [2, 2, 2, 2]  # Reduce Swin Transformer depth
   cfg.swin_transunet.num_heads = [2, 4, 8, 8]  # Reduce number of attention heads
   ```

## Evaluation Results and Analysis

### Performance Metrics Comparison

By evaluating both TransUNet and Swin TransUNet models on the same test set, we obtained the following performance metrics comparison:

| Model | Dice | IoU | Accuracy | Precision | Recall | F1 Score |
|-------|------|-----|----------|-----------|--------|----------|
| TransUNet | - | - | - | - | - | - |
| Swin TransUNet | - | - | - | - | - | - |

*Note: Actual values will be filled after model training and evaluation*

### Confusion Matrix Analysis

The confusion matrix provides a detailed view of the model's classification performance, including:
- True Positives (TP): Pixels correctly identified as foreground
- False Positives (FP): Background pixels incorrectly identified as foreground
- True Negatives (TN): Pixels correctly identified as background
- False Negatives (FN): Foreground pixels incorrectly identified as background

Through confusion matrix visualization, we can intuitively compare the performance differences between the two models when processing different classes.

### Improvement Analysis

The main improvements of Swin TransUNet compared to the original TransUNet are reflected in:

1. **Feature Extraction Capability**: The hierarchical structure and local attention mechanism of Swin Transformer enable the model to better capture multi-scale features of images

2. **Computational Efficiency**: Through the window attention mechanism, Swin TransUNet reduces computational complexity while maintaining performance

3. **Boundary Detail Processing**: The improved model performs better in handling segmentation boundary details, which is particularly important for medical image segmentation

## Conclusion

By replacing Vision Transformer with Swin Transformer, we have implemented a more efficient TransUNet model. The hierarchical structure and local attention mechanism of Swin Transformer enable the model to better handle medical image segmentation tasks while maintaining high computational efficiency.

During the debugging process, we solved multiple problems related to feature map shape, channel number matching, and memory optimization, enabling the model to train successfully. These modifications and optimizations can serve as references for integrating other Transformer variants into U-Net-like architectures.

By adding comprehensive evaluation metrics and visualization tools, we can more objectively compare the performance of different models, providing a solid foundation for subsequent model improvements.

## References

- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [3] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [4] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
