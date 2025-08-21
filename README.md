### ChestXRay-Pneumonia-Classification

Detect pneumonia from chest X-ray images using deep learning with TensorFlow/Keras. This repository contains multiple experiment notebooks exploring baseline CNNs and transfer learning architectures (ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121) with data augmentation and hyperparameter variations (batch sizes 4–32, fixed target size 180×180, binary classification).

## Key Features
- **Binary classification**: Pneumonia vs. Normal from chest X-rays
- **Transfer learning**: ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121
- **Data augmentation**: Rotation, flips, zoom, and shifts via `ImageDataGenerator`
- **Reproducible experiments**: Multiple phase notebooks varying batch size and steps-per-epoch
- **Metrics**: Accuracy, Precision, Recall tracked during training

## Dataset
Chest X-Ray dataset organized in directory structure for training/validation/test. See dataset link in `Dataset Link.txt`.

Expected structure:
```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```
Images are resized to `180x180`, `class_mode='binary'`.

## Environment Setup
Create a Python environment (Anaconda recommended) with TensorFlow/Keras and common imaging utilities.

### Dependencies
- Python 3.9+
- TensorFlow 2.x (tested with TF 2.x per notebook logs)
- Keras (bundled with TF 2.x)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (metrics/utilities)
- Jupyter Notebook

### Quick Start (Conda)
```bash
conda create -n chestxray python=3.9 -y
conda activate chestxray
pip install tensorflow==2.12.*
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

If using NVIDIA GPU, install CUDA/cuDNN versions compatible with your TensorFlow build and then `pip install tensorflow-gpu==2.12.*` (or the TF version matching your CUDA).

## Project Layout
```
Code(x21208590)/Code/
  phase1-(batch 4).ipynb
  phase1-(batch 8 ).ipynb
  phase1-(batch 16).ipynb
  phase1-(batch 4 steps 100).ipynb
  phase1-(batch 8 steps 100 ).ipynb
  phase1-(batch 16 steps 100).ipynb
  phase1(batch= 32).ipynb
  phase2-(batch 4 steps 100 DA).ipynb
  phase2-(batch 8  DA).ipynb
  phase2-(batch 8 steps 100 DA).ipynb
  phase2-(batch 8 steps 100 DA inc).ipynb
  phase2(batch 4 DA).ipynb
  Dataset Link.txt
```

## Usage
1) Download the dataset using the link in `Dataset Link.txt`, and arrange it into `data/train`, `data/val`, and `data/test` folders as shown above.

2) Open any notebook in Jupyter:
```bash
jupyter notebook
```

3) Update the following variables in the first cells as needed:
- `train_dir`, `val_dir`, `test_dir` → paths to your dataset folders
- `batch_size` → e.g., 4, 8, 16, or 32
- `target_size=(180, 180)` → keep consistent with model input

4) Run all cells to train and evaluate. Models compile with Adam optimizer and track Accuracy, Precision, and Recall.

### Minimal Example (generator setup excerpt)
```python
from keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(rotation_range=20,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     rescale=1./255)

train = image_generator.flow_from_directory(train_dir,
                                            target_size=(180, 180),
                                            batch_size=8,
                                            class_mode='binary')
```

## Results (example)
- Batch sizes evaluated: 4, 8, 16, 32
- Transfer models compared: ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121
- Metrics observed: Accuracy, Precision, Recall (see notebook outputs for exact numbers)

## Reproducibility Notes
- If you encounter "input ran out of data" warnings, adjust `steps_per_epoch` to match dataset size or use `.repeat()` with tf.data pipelines.
- For GPU errors in DenseNet/Conv2D layers, confirm compatible CUDA/cuDNN and TF versions.

## Contribution Guidelines
1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Run notebooks end-to-end to ensure changes don’t break training
3. Use clear commit messages and include before/after metrics or plots
4. Open a pull request describing your changes, motivation, and results

## License
This project is for academic/research use. If you plan to use it commercially, please open an issue to discuss licensing.


