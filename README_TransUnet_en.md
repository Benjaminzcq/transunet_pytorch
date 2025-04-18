# TransUNet: Transformer-Based Medical Image Segmentation Model
The unofficial implementation of [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306) on Pytorch

*This document provides detailed information about the implementation, training, and usage of the TransUNet model. TransUNet is a medical image segmentation model that combines CNN and Transformer, leveraging both CNN's local feature extraction capability and Transformer's global context modeling ability.*

![Output](./assets/outs.png "Output")
*Output of TransUNet implementation. (A) Original X-Ray Image; (B) Merged Image of the Predicted Segmentation Map and Original X-Ray; (C) Ground Truth; (D) Predicted Segmentation Map*

## Model Overview

- TransUNet demonstrates excellent performance in various medical image segmentation tasks, especially when dealing with complex medical images that require global context understanding.
- The model adopts a U-shaped architecture, combining high-resolution spatial information from CNN features and global context encoded by Transformers.
- Compared to traditional U-Net, TransUNet can better handle long-range dependencies, improving segmentation accuracy.

## Model Architecture
![Model Architecture](./assets/arch.png "Model Architecture")

*TransUNet Architecture Figure from Official Paper*

TransUNet's main components include:
1. **CNN Encoder**: Uses ResNet as the backbone network for feature extraction
2. **Transformer Encoder**: Converts features extracted by CNN into sequences and captures global dependencies through multi-head self-attention mechanisms
3. **Decoder**: Restores spatial details through upsampling and skip connections to generate the final segmentation map

## Dependencies

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

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

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

## Model Training

Train the TransUNet model using the following command:

```bash
python train_transunet.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
```

Parameter description:
- `--train_path`: Training dataset path, default is "./DRIVE/training"
- `--test_path`: Testing dataset path, default is "./DRIVE/test"
- `--model_path`: Model saving path, default is "./model_transunet.pth"

Example:
```bash
python train_transunet.py --train_path ./DRIVE/training --test_path ./DRIVE/test --model_path ./model_transunet.pth
```

## Model Inference

Perform image segmentation prediction using the trained model:

```bash
python inference_transunet.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
```

Parameter description:
- `--model_path`: Model weights path, default is "./model_transunet.pth"
- `--image_path`: Input image path, can be a single image or a folder containing multiple images, default is "./DRIVE/test/images"
- `--output_folder`: Output folder name, default is "transunet_results"
- `--merged`: Whether to merge segmentation results with the original image, default is True
- `--no_save`: Set this parameter to not save results

Example:
```bash
python inference_transunet.py --model_path ./model_transunet.pth --image_path ./DRIVE/test/images --output_folder transunet_results
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
python evaluate_models.py --test_path ./DRIVE/test --transunet_model ./model_transunet.pth --output_dir ./evaluation_results
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
3. Performance comparison chart (if Swin TransUNet is also evaluated)
4. CSV file containing all metrics

## Model Parameter Adjustment

If you encounter memory issues, try the following adjustments:

1. Reduce batch size: Modify the batch_size parameter in config.py
   ```python
   cfg.batch_size = 2  # Reduce batch size to decrease memory usage
   ```

2. Lower model complexity: Modify TransUNet configuration in config.py
   ```python
   cfg.transunet.out_channels = 64  # Reduce output channels
   cfg.transunet.mlp_dim = 256      # Reduce MLP dimension
   ```

## References

- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [3] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
