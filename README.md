# Video Classification using HDBM (CNN-RNN and 3DResNet)

This repository implements video classification models using HDBM (Hidden Dynamic Block Model). Two different approaches are explored:
1. **CNN-RNN model**
2. **3DResNet model**

## Overview

This project aims to classify videos using deep learning models. The models used in this repository are built to handle video data and extract meaningful features using convolutional and recurrent layers (CNN-RNN), as well as 3D convolutions in the case of 3DResNet.

## Features

- **Dataset handling**: Code to load and preprocess the video dataset.
- **Frame extraction**: Extracts frames from videos for model training.
- **Model implementation**:
  - CNN-RNN: A combination of Convolutional Neural Networks and Recurrent Neural Networks for sequential video data.
  - 3DResNet: A variant of ResNet that operates on 3D video data for classification tasks.
- **Inference**: Run inference on a trained model to classify new videos.
- **Training script**: Code to train the models on the dataset.

## Files

- **dataset.py**: Handles dataset loading and preprocessing.
- **extract_frames.py**: Extracts frames from video files for model input.
- **inference.py**: Contains functions to perform inference on the trained model.
- **model.py**: Contains the architecture of the CNN-RNN and 3DResNet models.
- **train.py**: Script to train the model on the given dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Mjavan/HMDB-video-classification.git
    ```

2. Install dependencies:
    - If using **`pip`**:
      ```bash
      pip install -r requirements.txt
      ```
    - If using **`conda`**:
      ```bash
      conda create --name video-classification-env --file requirements.txt
      ```

3. (Optional) Set up a virtual environment (recommended):
    - For **virtualenv**:
      ```bash
      python3 -m venv env
      source env/bin/activate   # On Windows use `env\Scripts\activate`
      ```

## Usage
1. **Download data** from:
    [this Kaggle link](https://www.kaggle.com/datasets/ashrafmahmud/test-hmdb-51?select=train_1).
   After downloading, save the training data in the `./data/train_1` directory and the test data in the `./data/test` directory.

2. **Extract frames** from your video dataset:
    ```bash
    python extract_frames.py 
    ```

3. **Train the model**:
    ```bash
    python train.py --epochs 10 --batch_size 32 --model_type cnn_rnn
    ```

4. **Run inference** to classify a video:
    ```bash
    python inference.py --model_type cnn_rnn --root /path/to/saved_model
    ```
    You should obtain 94% accuracy on validation set using 3DResNet and 90% accuracy using CNN-RNN. 
## Models

This repository includes two models:

- **CNN-RNN**: A combination of CNN for feature extraction and RNN for sequence modeling.
- **3DResNet**: A 3D convolutional network for video classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **HDBM**: The concept of Hidden Dynamic Block Model (HDBM) for video classification.
- **PyTorch**: Used for building and training the models.
- **OpenCV**: Used for video frame extraction.

