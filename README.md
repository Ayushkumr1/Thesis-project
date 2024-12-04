# Analyzing the Impact of Deep Learning and Data Augmentation on Medical Image Classification


This research investigates the role of deep learning techniques and data augmentation in improving the performance of models for medical image classification. The study focuses on identifying pneumonia in pediatric patients using frontal chest X-ray images.

Objectives
The primary goal is to evaluate how hyperparameter tuning and data augmentation affect the performance of pre-trained Convolutional Neural Network (CNN) models for medical image classification.

Dataset
The dataset comprises 5,863 chest X-ray images, categorized into two classes: Normal and Pneumonia. Images were sourced from pediatric patients aged 1-5 years and divided into training, testing, and validation sets.

Methodology
Models: Five pre-trained CNN architectures were used:
ResNet50
EfficientNetB0
VGG16
InceptionV3
DenseNet121
Phases:
Phase 1: Models were trained and tested without data augmentation.
Phase 2: Data augmentation techniques (e.g., random rotation, zoom, and normalization) were applied to assess their impact on performance.
Evaluation Metrics: Accuracy, Precision, Recall, Dice Score, and Binary Cross-Entropy Loss were used to measure performance.
Findings
Data augmentation generally enhanced performance by reducing cross-entropy loss and increasing overall accuracy.
VGG16 demonstrated robust performance, particularly in recall, when trained on smaller batch sizes (4 and 8).
Computational limitations caused training errors for larger batch sizes due to insufficient GPU memory.
The dataset's bias (74% pneumonia cases) impacted model generalization, emphasizing the need for diverse data.
Conclusion
The study highlights the importance of data augmentation and hyperparameter tuning in medical image classification. While data augmentation significantly improved some metrics, it occasionally reduced precision and recall. Future work aims to explore advanced augmentation and optimization techniques to develop more robust and reliable models.
