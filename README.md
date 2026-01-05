This project implements a Convolutional Neural Network (CNN) using MATLAB to classify skin lesion into 7 categories:

nv (Melanocytic nevi)
mel (Melanoma)
bkl (Benign keratosis)
bcc (Basal cell carcinoma)
akiec (Actinic keratoses)
vasc (Vascular lesions)
df (Dermatofibroma)

Dataset:
I used HAM10000 dataset from kaggle, it is free via this link:
Download link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Extract the folder an run sorting script, the sorting script:

Reads image labels from HAM10000_metadata.csv
Matches image IDs to lesion types
Organizes images into class-based directories required by MATLAB’s imageDatastore

The images will be organized as follows:

data/
 ├── nv/
 ├── mel/
 ├── bkl/
 ├── bcc/
 ├── akiec/
 ├── vasc/
 └── df/

CNN Architecture:
Input size: 224 × 224 × 3 (RGB images)
4 convolutional blocks
Each block has Convolution → Batch Normalization → ReLU → Max Pooling
Filter sizes increase progressively (32 → 64 → 128 → 256)
Global Average Pooling to reduce parameters
Dropout (for global pooling) to prevent overfitting
Fully connected layer with 7 neurons (one per class)
Softmax layer (produces probability distribution) 
Classification layer for final prediction

