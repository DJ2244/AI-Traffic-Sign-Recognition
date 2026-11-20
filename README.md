# AI based road object detection for autonomous vehicles

## project overview

This project investigtes the use of AI and sdeep learning within road images. These include vehicles, pedestrians, traffic signs and other road related objects. 

## Technologies used

- python
- YOLOv8
- Ultralytics
- OpenCV
- Matplotlib
- PyCharm

## Dataset

The project uses the road object detection dataset obtained from kaggle. 

The Dataset contains 25 object categories and is divided into training, validation and testing subsets. 

dataset source:
https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset

## model training

The YOLOv8 was used as the starting model. 

Training configuration:
- Epochs: 5
- Training data fraction: 10%
- Image size: 640
- Batch size: 8 
- Number of classes: 25

## Model evaluation

The trained model was evaluated using the test dataset.

Results:

| Metric | Result |
|---|---:|
| Precision | 70.7% |
| Recall | 67.6% |
| mAP50 | 71.9% |
| mAP50-95 | 51.2% |

## Installation

Clone this repository
```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the Python scripts using PyCharm or a terminal.

Main scripts include:

- `main.py` - Loads the YOLOv8 model.
- `train.py` - Trains the object detection model.
- `test_trained_model.py` - Tests the trained model.
- `evaluate_model.py` - Evaluates model performance.
- `compare_models.py` - Compares the original and trained models.

Ensure the dataset and trained model weights are
available in the required project folders before running
the scripts.

## Project Aim

The aim is to explore how AI-based object detection
could contribute to the development of autonomous
vehicles and improve the recognition of objects
within road environments.