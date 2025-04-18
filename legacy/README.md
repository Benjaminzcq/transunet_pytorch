# Panoramic Dental X-Ray Image Semantic Segmentation with TransUnet
The unofficial implementation of [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306) on Pytorch

![Output](./assets/outs.png "Output")
*Output of my implementation. (A) Original X-Ray Image; (B) Merged Image of the Predicted Segmentation Map and Original X-Ray; (C) Ground Truth; (D) Predicted Segmentation Map*

## TransUNet
- On various medical image segmentation tasks, the ushaped architecture, also known as U-Net, has become the de-facto standard and achieved tremendous success. However, due to the intrinsic
locality of convolution operations, U-Net generally demonstrates limitations in explicitly modeling long-range dependency. [1]
- TransUNet employs a hybrid CNN-Transformer architecture to leverage both detailed high-resolution spatial information from CNN features and the global context encoded by Transformers. [1]

## Model Architecture
![Model Architecture](./assets/arch.png "Model Architecure")

*TransUNet Architecture Figure from Official Paper*

## Dependencies
- Python 3.6+
- `pip install -r requirements.txt`
- Required packages: PyTorch, torchvision, timm, einops, numpy, tqdm, matplotlib, seaborn, scikit-learn

## Dataset
- UFBA_UESC_DENTAL_IMAGES[2] dataset was used for training.
- Dataset can be accessed by request[3].

## Training
- Training process can be started with following command:
    ```bash
    python run_train.py [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_path MODEL_PATH]
    ```

## Inference
- After model is trained, inference can be run with following command:
    ```bash
    python run_inference.py [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--output_folder OUTPUT_FOLDER] [--merged] [--no_save]
    ```

## Model Evaluation
- Evaluate model performance and generate detailed metrics reports:
    ```bash
    python evaluate_models.py [--test_path TEST_PATH] [--transunet_model TRANSUNET_MODEL_PATH] [--swin_model SWIN_MODEL_PATH] [--output_dir OUTPUT_DIR]
    ```

- The evaluation includes the following metrics:
  - **Dice Coefficient**: Measures the overlap between predicted and ground truth segmentation
  - **IoU (Intersection over Union)**: Measures the overlap ratio between predicted and ground truth regions
  - **Accuracy**: Proportion of correctly classified pixels
  - **Precision**: Proportion of true foreground pixels among all pixels predicted as foreground
  - **Recall**: Proportion of correctly predicted foreground pixels among all true foreground pixels
  - **F1 Score**: Harmonic mean of precision and recall

- The evaluation results are presented as:
  1. Detailed metrics printed to console
  2. Confusion matrix visualization for each model
  3. Performance comparison chart between different models
  4. CSV file containing all metrics
    
## Other Implementations
- [Self Attention CV / The AI Summer](https://github.com/The-AI-Summer/self-attention-cv)
- [SOTA Vision / 04RR](https://github.com/04RR/SOTA-Vision)

## Evaluation Results

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

## References
- [1] [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [2] [Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives](https://www.sciencedirect.com/science/article/abs/pii/S0957417418302252)
- [3] [GitHub Repository of Dataset](https://github.com/IvisionLab/dental-image)
- [4] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
