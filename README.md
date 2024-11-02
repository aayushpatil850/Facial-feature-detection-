# Facial-feature-detection-
Here's a well-organized README file in markdown format for your GitHub project:

---

# Arched Eyebrows Detection with ResNet18

This project implements a deep learning pipeline for detecting the presence of arched eyebrows in images using a fine-tuned ResNet18 model. The model is trained on a custom dataset of facial images, and leverages PyTorch for training, evaluation, and experimentation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Occlusion Sensitivity Analysis](#occlusion-sensitivity-analysis)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Overview
This project uses a transfer learning approach with ResNet18 to classify whether a person has arched eyebrows in an image. The dataset is loaded, preprocessed, and augmented using PyTorch's `Dataset` and `DataLoader`. Early stopping and learning rate scheduling are implemented for efficient training.

## Dataset
The dataset contains facial images with labels indicating the presence or absence of arched eyebrows. The images are stored in a directory, and their corresponding labels are in a CSV file.

1. **Images Directory**: `/path/to/images`
2. **CSV File**: `face_image_attr.csv`

The CSV file should have the following format:

| image_name | attribute_1 | Arched_Eyebrows |
|------------|-------------|------------------|
| img_001.jpg | value      | 1               |

The project is structured to load images, resize to 224x224, normalize, and split into training, validation, and test sets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arched-eyebrows-detection.git
   cd arched-eyebrows-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
The ResNet18 model is pre-trained on ImageNet and fine-tuned on our dataset. The training loop includes early stopping and a learning rate scheduler for better convergence.

### Run Training
```python
python train.py
```

The training script performs the following:
- Freezes all ResNet layers except the final fully connected layer.
- Trains with Binary Cross-Entropy loss and optimizes with Adam.
- Saves the best model based on validation loss.

### Parameters
To experiment with different parameters, modify:
- Learning rate (`lr`)
- Batch size (`batch_size`)
- Number of epochs (`epochs`)

## Evaluation
The evaluation metrics include precision, accuracy, F1 score, and ROC AUC. A confusion matrix is also generated.

```python
python evaluate.py
```

### Example Output
```
Precision: 0.87
Accuracy: 0.91
F1 Score: 0.89
ROC AUC: 0.94
```

## Experiments
The `run_experiment` function automates hyperparameter tuning for various combinations of learning rate, batch size, and epochs. Each experiment is logged and results are summarized at the end.

## Occlusion Sensitivity Analysis
Occlusion sensitivity analysis helps visualize which parts of the image influence the model’s decision. A heatmap is generated to highlight regions that contribute most to the prediction.

```python
python occlusion_analysis.py
```

## Visualization
- **Filter Visualization**: Visualizes the filters learned in each ResNet layer.
- **Activation Maps**: Displays activation maps from selected ResNet layers.

The `visualization.py` script captures and plots activation maps and filters.

## Results
Final results, including the classification accuracy and ROC AUC, are presented with heatmaps and ROC curves.




---

